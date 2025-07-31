# 🦄 Next Steps Action Plan

## Current Status
✅ **INT4 GEMM confirmed**: 7.5x speedup (52.4 GFLOPS effective)
❌ **Projected performance**: 0.62 tok/s (still 34x away from 21 tok/s target)
⚠️ **Issue identified**: Low base GFLOPS (7.0) indicates kernel optimization needed

## Immediate Actions (Today)

### 1. Debug Low GFLOPS Performance
The FP32 baseline shows only 7.0 GFLOPS, which is extremely low for gfx1103:
- **Expected**: 200+ GFLOPS
- **Actual**: 7.0 GFLOPS
- **Problem**: Poor kernel optimization or launch configuration

Let's investigate:
```bash
# Check if we're using the right work group sizes
clinfo | grep -A5 "Max work group size"

# Test with optimized kernel
python3.13 magic_unicorn_opencl_int4.py
```

### 2. Run Full OpenCL INT4 Engine Test
Test the complete implementation with proper optimizations:
```bash
python3.13 magic_unicorn_opencl_int4.py
```

This includes:
- Blocked GEMM kernels
- Proper work group sizing
- Memory coalescing

### 3. Benchmark with Existing Optimized Pipeline
Compare with our proven OpenCL implementation:
```bash
python3.13 optimized_hybrid_pipeline.py
```

Get baseline performance to compare against.

## Root Cause Analysis

### Why Only 0.62 tok/s?

1. **Kernel Launch Overhead**: Simple kernel has high overhead
2. **No Memory Optimization**: Test uses naive memory access
3. **No Blocking**: Missing register/shared memory blocking
4. **Small Matrices**: 2048x2048 doesn't fully utilize GPU

### Expected Performance with Optimizations

| Optimization | Impact | Cumulative |
|--------------|--------|------------|
| Blocked kernel | 10x | 6.2 tok/s |
| Proper work groups | 2x | 12.4 tok/s |
| Memory coalescing | 1.5x | 18.6 tok/s |
| NPU attention | 1.2x | 22.3 tok/s ✓ |

## Action Plan

### Step 1: Optimize OpenCL Kernels (Today)
1. Test `magic_unicorn_opencl_int4.py` with blocked kernels
2. Profile with `rocprof` to identify bottlenecks
3. Tune work group sizes for gfx1103

### Step 2: Integration Testing (Today/Tomorrow)
1. Integrate INT4 kernels into `optimized_hybrid_pipeline.py`
2. Test with real model shapes (2560 hidden size)
3. Benchmark full transformer layer

### Step 3: NPU Integration (This Week)
1. Use proven NPU memory allocation
2. Implement simple attention kernel
3. Pipeline NPU+iGPU execution

### Step 4: Final Optimizations (This Week)
1. Kernel fusion (QKV, activation+projection)
2. Memory layout optimization
3. Multi-stream execution

## Commands to Run Now

```bash
# 1. Test optimized INT4 implementation
python3.13 magic_unicorn_opencl_int4.py

# 2. Benchmark current best implementation
python3.13 optimized_hybrid_pipeline.py

# 3. Quick benchmark with CLI
./unicorn_benchmark_suite.py --quick --device igpu --quantization int4

# 4. Check GPU utilization
rocm-smi --showuse
```

## Success Metrics

We need to see:
1. **GEMM Performance**: 200+ GFLOPS (not 7)
2. **Layer Time**: <6ms (not 38ms)
3. **Tokens/sec**: 4+ (not 0.62)

With proper optimization, INT4 alone should get us to 4-5 tok/s, then NPU and other optimizations push us over 21 tok/s.

## If Performance Still Low

Alternative approaches:
1. **Use INT8**: Better hardware support
2. **Try Vulkan**: Native AMD driver stack
3. **Direct Assembly**: Maximum performance
4. **Reduce Context**: Test with shorter sequences

The key insight remains valid - INT4 provides the memory bandwidth reduction we need. We just need proper kernel optimization to realize the computational benefits.

Let's start with testing the optimized implementation! 🚀