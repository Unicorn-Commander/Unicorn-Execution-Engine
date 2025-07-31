# 🦄 Fallback Optimization Strategy

## Context
Gemini encountered ROCm compilation issues with the HIP WMMA kernels. This document outlines alternative approaches to achieve the 21 tok/s target without requiring ROCm's rocWMMA library.

## Current Situation
- **Baseline**: 0.366 tok/s (FP32 OpenCL)
- **Target**: 21 tok/s (57.4x improvement needed)
- **Blocker**: ROCm HIP/rocWMMA compilation fails

## Alternative Approaches

### Option 1: OpenCL INT4 Implementation (Recommended)
**Implementation**: `magic_unicorn_opencl_int4.py`

Advantages:
- Works with existing OpenCL infrastructure
- No ROCm dependencies
- Can still achieve significant speedup

Expected performance:
- INT4 memory reduction: 8x
- Compute speedup: 3-5x (limited by lack of native INT4 ops)
- Overall speedup: 4-6x

### Option 2: Vulkan Compute Shaders
AMD RDNA3 has excellent Vulkan support with potential INT4 operations:

```python
# Use Kompute or native Vulkan
import kp  # kompute

manager = kp.Manager()
shader = """
#version 450
// INT4 compute shader
layout(local_size_x = 256) in;
// Vulkan supports packed formats
"""
```

Advantages:
- Native AMD driver support
- Potential access to hardware INT4 ops
- Cross-platform

### Option 3: Direct Assembly via CLRadeonExtender
Use AMD's assembly tools to write native GCN/RDNA assembly:

```bash
# Install CLRadeonExtender
git clone https://github.com/CLRX/CLRX-mirror.git

# Write assembly kernel
.kernel gemm_int4_native
    .config
        .dims xyz
        .cws 16, 16, 1
    .text
        # Direct RDNA3 assembly for INT4 ops
        v_dot4_i32_i8 v0, s0, s1, v0  # Native INT8 dot product
```

Advantages:
- Direct hardware access
- Can use undocumented instructions
- Maximum performance

### Option 4: Mixed Precision Approach
Instead of pure INT4, use a mixed approach:

1. **Critical layers**: Keep FP16/FP32
2. **Bulk layers**: Use INT8 (better OpenCL support)
3. **Attention**: Optimize separately

```python
def mixed_precision_forward(x, layer_idx):
    if layer_idx < 5:  # First layers more sensitive
        return forward_fp16(x)
    elif layer_idx < 35:  # Middle layers
        return forward_int8(x)
    else:  # Final layers
        return forward_fp32(x)
```

## Immediate Action Plan

### Step 1: Test OpenCL INT4 Concept
```bash
python3.13 test_opencl_int4_simple.py
```

Expected output:
- Verify 3-5x speedup achievable
- Check memory bandwidth utilization
- Validate INT4 accuracy

### Step 2: Optimize OpenCL Kernels
Focus areas:
1. **Register blocking**: 4x4 or 8x8 tiles
2. **Shared memory**: Reduce global memory access
3. **Vectorization**: Use vload/vstore for INT4
4. **Workgroup optimization**: Tune for RDNA3

### Step 3: Implement Hybrid Approach
Combine multiple optimizations:

| Component | Optimization | Expected Speedup |
|-----------|-------------|------------------|
| GEMM | OpenCL INT4 | 4x |
| Attention | NPU offload | 2x |
| Memory | Zero-copy buffers | 1.5x |
| Pipeline | Kernel fusion | 1.5x |
| **Total** | **Combined** | **18x** |

### Step 4: Final Push to 21 tok/s
Additional optimizations:
- Dynamic batching for better GPU utilization
- Activation quantization (not just weights)
- Profile-guided optimization
- CPU-GPU overlap

## Performance Projections

### Conservative (OpenCL INT4 only)
- Current: 0.366 tok/s
- With INT4: 1.5-2.2 tok/s (4-6x)
- Gap to target: 10-14x

### Realistic (Hybrid approach)
- INT4 GEMM: 4x
- NPU attention: 2x  
- Optimizations: 2.5x
- **Total: 20x → 7.3 tok/s**

### Optimistic (All optimizations)
- INT4 + vectorization: 6x
- NPU + pipelining: 3x
- Memory + fusion: 3x
- **Total: 54x → 19.8 tok/s** ✓

## Recommended Path Forward

1. **Validate OpenCL INT4** (Today)
   - Run test_opencl_int4_simple.py
   - Measure actual speedup
   - Identify bottlenecks

2. **Optimize Kernels** (This week)
   - Implement register blocking
   - Add shared memory tiling
   - Profile and tune

3. **Integrate NPU** (Next week)
   - Use proven NPU memory allocation
   - Implement attention kernel
   - Pipeline with iGPU

4. **Final Optimizations** (Following week)
   - Kernel fusion
   - Memory optimizations
   - Performance validation

## Fallback if INT4 Insufficient

If OpenCL INT4 doesn't provide enough speedup:

1. **Try INT8**: Better hardware support, still 4x memory reduction
2. **Use FP16**: 2x memory reduction, native hardware support
3. **Optimize algorithm**: Flash Attention, sparse attention patterns
4. **Reduce model**: Distillation, pruning, smaller architecture

## Conclusion

While the ideal HIP WMMA approach is blocked, we have multiple viable paths to reach 21 tok/s. The OpenCL INT4 implementation combined with NPU integration and optimization techniques should achieve our target. The key is systematic optimization and measurement at each step.

**Next immediate action**: Run the OpenCL INT4 test to validate our performance projections. 🦄⚡