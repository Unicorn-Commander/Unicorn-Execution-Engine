# 🔍 Performance Analysis & Optimization Plan

## Current Performance Gap

### Measured Results
| Implementation | Seq=32 | vs Baseline | vs Target |
|----------------|--------|-------------|-----------|
| FP32 Baseline | 3.5 tok/s | 1.0x | 6.0x |
| INT4 Current | 1.35 tok/s | 0.39x | 15.5x |
| Target | 21 tok/s | 6.0x | 1.0x |

**Problem**: INT4 is currently 2.6x SLOWER than FP32 baseline!

## Root Cause Analysis

### 1. Kernel Configuration Issues
- **Work group size**: Only 8x8 = 64 threads (should be 256)
- **Register blocking**: 2x2 is too small for RDNA3
- **Tile size**: 16x16 doesn't match hardware (should be 32x32 or 64x64)

### 2. Memory Access Patterns
- **INT4 unpacking**: Done in inner loop (huge overhead)
- **Scale loading**: Per-row instead of per-matrix
- **No vectorization**: Loading single bytes instead of vectors

### 3. Algorithm Issues
- **No prefetching**: Stalls on memory loads
- **Poor occupancy**: Not enough threads to hide latency
- **Attention on CPU**: Should use iGPU

## Optimization Strategy

### Phase 1: Fix Critical Issues (2-3x speedup)
```opencl
// Better work group configuration
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREADS 256

// Vectorized INT4 loading
uint4 packed = vload4(0, A_packed + offset);
// Unpack 32 INT4 values at once
```

### Phase 2: Memory Optimization (2x speedup)
- Pre-unpack INT4 weights once
- Use texture memory for weights
- Implement double buffering

### Phase 3: Algorithm Optimization (2x speedup)
- Fuse operations (QKV in single kernel)
- Implement Flash Attention variant
- Use NPU for attention heads

## Immediate Fix

Let's create an optimized version that addresses the critical issues:

### Step 1: Better Kernel Launch Configuration
```python
# Current (poor)
global_size = ((N + 15) // 16 * 8, (M + 15) // 16 * 8)
local_size = (8, 8)  # Only 64 threads!

# Optimized
global_size = ((N + 63) // 64 * 16, (M + 63) // 64 * 16) 
local_size = (16, 16)  # 256 threads
```

### Step 2: Pre-unpack Weights
Instead of unpacking in kernel, do it once during quantization:
```python
def quantize_and_unpack_int4(weights):
    # Quantize to INT4
    # Unpack to INT8 for faster kernel access
    # Store scales separately
```

### Step 3: Use Proven Kernels
Adapt the working kernels from `optimized_hybrid_pipeline.py` with INT4:
- Already achieves 3.5 tok/s with FP32
- Just need to add INT8/INT4 support

## Expected Performance After Optimization

| Optimization | Impact | Cumulative | Performance |
|--------------|--------|------------|-------------|
| Current INT4 | - | 1x | 1.35 tok/s |
| Fix work groups | 2x | 2x | 2.7 tok/s |
| Vectorization | 2x | 4x | 5.4 tok/s |
| Pre-unpacking | 1.5x | 6x | 8.1 tok/s |
| Kernel fusion | 1.5x | 9x | 12.2 tok/s |
| NPU attention | 1.7x | 15.3x | 20.7 tok/s ✓ |

## Action Items

1. **Quick Fix** (Today):
   - Modify work group sizes in existing kernel
   - Pre-unpack INT4 to INT8
   - Test performance

2. **Proper Solution** (This Week):
   - Port optimized kernels from hybrid pipeline
   - Add INT8 support (better than INT4 unpacking)
   - Implement kernel fusion

3. **Final Push** (Next Week):
   - NPU attention integration
   - Profile-guided optimization
   - Production deployment

## Alternative: Use INT8 Instead

Given the INT4 unpacking overhead, INT8 might be better:
- Native hardware support
- No unpacking needed  
- Still 4x memory reduction
- Proven to work in other frameworks

The key insight remains: **Quantization is essential**, but INT8 might be more practical than INT4 for this hardware.

## Next Command

Let's test with INT8 to see if it performs better:
```bash
# Modify the test to use INT8
sed -i 's/int4/int8/g' test_opencl_int4_simple.py
sed -i 's/INT4/INT8/g' test_opencl_int4_simple.py
sed -i 's/& 0xF/& 0xFF/g' test_opencl_int4_simple.py
python3.13 test_opencl_int8_simple.py
```