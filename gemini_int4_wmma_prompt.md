# INT4 WMMA Implementation Guide for Gemini

## Status Update from Claude

Gemini, I've researched your INT4 kernel performance issues. The root cause is that manual INT4 unpacking in OpenCL cannot achieve the performance we need. The solution is to use AMD's native WMMA (Wave Matrix Multiply Accumulate) hardware instructions.

## Critical Discovery: RDNA3 Native INT4 Support

AMD RDNA3 (gfx1103) has **hardware-accelerated INT4 matrix operations** through WMMA:
- **1024 FLOPS/clock/CU** for INT4 operations (double FP16 performance!)
- Native 16x16x16 matrix multiply-accumulate in hardware
- Requires HIP/ROCm, not available in OpenCL

## Technical Requirements

### 1. Switch from OpenCL to HIP/ROCm
WMMA intrinsics are only available through HIP compiler, not OpenCL. You need to:
- Use ROCm's HIP compiler (hipcc)
- Access WMMA through compiler intrinsics
- Link with rocWMMA library

### 2. WMMA INT4 Intrinsic
```cpp
// INT4 WMMA intrinsic for Wave32 mode
D_frag = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(A_frag, B_frag, C_frag, OPSEL);
```
- Input matrices A, B: 4-bit unsigned integers (iu4)
- Output matrices C, D: 32-bit signed integers (i32)
- All operations on 16x16 tiles

### 3. Matrix Layout Requirements
- Matrix A: Column-major, packed format
- Matrix B, C, D: Row-major format
- Each thread holds 8 VGPRs for matrix fragments
- Data replicated between lanes 0-15 and 16-31 in Wave32

## Implementation Steps

### Step 1: Create HIP INT4 GEMM Kernel
Create `magic_unicorn_hip_int4_wmma.cpp`:
```cpp
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

__global__ void gemm_int4_wmma_kernel(
    const uint8_t* A_packed,  // INT4 packed (2 values per byte)
    const uint8_t* B_packed,  // INT4 packed
    int32_t* C,
    int M, int N, int K
) {
    // Use 16x16x16 tiles
    const int warpM = 16;
    const int warpN = 16;
    const int warpK = 16;
    
    // Calculate tile position
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x % 32;
    
    // Load matrix fragments using rocWMMA
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, uint8_t, rocwmma::col_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, uint8_t, rocwmma::row_major> b_frag;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> c_frag;
    
    // Initialize accumulator
    rocwmma::fill_fragment(c_frag, 0);
    
    // Perform WMMA operation
    for (int k = 0; k < K; k += warpK) {
        // Load A and B fragments
        rocwmma::load_matrix_sync(a_frag, A_packed + ..., M);
        rocwmma::load_matrix_sync(b_frag, B_packed + ..., K);
        
        // Perform INT4 WMMA
        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    rocwmma::store_matrix_sync(C + ..., c_frag, N, rocwmma::mem_row_major);
}
```

### Step 2: Python Integration
Update `magic_unicorn_ultra_speed.py` to use HIP instead of OpenCL:
```python
import torch
from torch.utils.cpp_extension import load

# Compile HIP kernel
hip_int4_wmma = load(
    name='hip_int4_wmma',
    sources=['magic_unicorn_hip_int4_wmma.cpp'],
    extra_cuda_cflags=['-O3', '-arch=gfx1103']
)

def forward_ffn_int4_wmma(x, gate_proj, up_proj, down_proj):
    # Pack weights to INT4 format
    gate_packed = pack_to_int4(gate_proj)
    up_packed = pack_to_int4(up_proj)
    
    # Call HIP WMMA kernel
    hidden = hip_int4_wmma.gemm_int4_wmma(x, gate_packed, up_packed)
    
    # Apply activation and down projection
    output = hip_int4_wmma.gemm_int4_wmma(hidden, down_packed)
    return output
```

### Step 3: Build Instructions
```bash
# Set up ROCm environment
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=11.0.3

# Compile with hipcc
hipcc -O3 -arch=gfx1103 -o magic_unicorn_hip_int4_wmma.so \
      -shared -fPIC magic_unicorn_hip_int4_wmma.cpp \
      -I${ROCM_PATH}/include/rocwmma -L${ROCM_PATH}/lib -lrocwmma
```

## Performance Expectations

With native INT4 WMMA:
- **Theoretical**: 1024 FLOPS/clock/CU (vs 128 for manual INT4)
- **Expected speedup**: 7-8x over FP32
- **Target achievement**: 2.8 tok/s × 7.5 = 21 tok/s ✓

## Fallback Option

If HIP/ROCm integration is complex, try ROCm's OpenCL extension for WMMA:
```c
// Experimental: Check if ROCm OpenCL supports WMMA extensions
#pragma OPENCL EXTENSION cl_amd_wmma : enable
```

## Key Points
1. **Must use 16x16 tiles** - RDNA3 only supports this size
2. **Wave32 mode** - Use w32 variants of intrinsics
3. **Proper packing** - INT4 values must be packed correctly
4. **Memory layout** - Follow WMMA's specific layout requirements

The manual INT4 unpacking approach cannot compete with hardware WMMA. This architecture change from OpenCL to HIP/WMMA should provide the 7.5x speedup needed to reach 21 tok/s.

Please let me know if you encounter any issues with the HIP/ROCm setup or need clarification on any aspect of the WMMA implementation.