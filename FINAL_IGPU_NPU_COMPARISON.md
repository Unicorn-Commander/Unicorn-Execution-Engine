# 🎯 Final iGPU vs NPU+iGPU Performance Comparison

## Executive Summary

Testing shows that the **iGPU-only and NPU+iGPU hybrid pipelines perform nearly identically**, with the hybrid approach showing only marginal improvements (0.7-0.9%) in most cases.

## 📊 Performance Results

### Tokens Per Second Comparison

| Sequence Length | iGPU-only | NPU+iGPU Hybrid | Improvement |
|-----------------|-----------|-----------------|-------------|
| 32 tokens | 5.2 tok/s | 5.0 tok/s | -4.4% |
| 128 tokens | 16.5 tok/s | 16.7 tok/s | +0.9% |
| 256 tokens | 22.7 tok/s | 22.9 tok/s | +0.7% |

### Operation-Level Timing (ms)

#### Small Context (32 tokens)
| Operation | iGPU | Hybrid | Notes |
|-----------|------|--------|-------|
| QKV Projections | 47.2 | 55.6 | iGPU 15% faster |
| Attention | 13.6 | 1.7 | Hybrid 8x faster (but small impact) |
| Output Projection | 14.9 | 18.5 | iGPU 20% faster |
| FFN | 78.1 | 78.9 | Nearly identical |
| **Total** | 153.8 | 154.8 | Essentially the same |

#### Medium Context (128 tokens)
| Operation | iGPU | Hybrid | Notes |
|-----------|------|--------|-------|
| QKV Projections | 50.6 | 49.3 | Hybrid 3% faster |
| Attention | 9.3 | 8.2 | Hybrid 13% faster |
| Output Projection | 17.6 | 16.7 | Hybrid 6% faster |
| FFN | 110.7 | 112.2 | Nearly identical |
| **Total** | 188.1 | 186.3 | Hybrid 1% faster |

## 🔍 Key Findings

### 1. **Minimal Performance Difference**
- The NPU+iGPU hybrid approach shows **less than 1% improvement** over iGPU-only
- In some cases (small context), the hybrid is actually slightly slower

### 2. **Attention Acceleration Limited Impact**
- While NPU attention is significantly faster (up to 8x), attention is only 5-10% of total compute
- The majority of time is spent in GEMM operations (QKV, FFN projections)

### 3. **iGPU Efficiency**
- The AMD iGPU (gfx1103) handles GEMM operations very efficiently
- OpenCL optimized kernels are well-suited for the workload
- 38GB of shared memory eliminates data transfer overhead

### 4. **Bottleneck Analysis**
- **Primary bottleneck**: Memory bandwidth (shared between CPU/GPU/NPU)
- **Secondary bottleneck**: GEMM operations dominate (70-80% of compute)
- **NPU impact limited**: Attention is only 5-10% of total compute time

## 💡 Recommendations

### For Maximum Performance:
1. **Use iGPU-only pipeline** - Simpler and performs nearly identically
2. **Focus on GEMM optimization** - This is where most time is spent
3. **Consider lower quantization** - Q4 models are 2.7x faster than Q8

### NPU Usage:
- NPU provides minimal benefit for current transformer architectures
- May be more beneficial for models with higher attention:FFN ratio
- Could be valuable for specialized attention mechanisms (sparse, flash attention)

### Optimization Priorities:
1. **Memory bandwidth** - Main limiting factor
2. **GEMM kernels** - Further optimize matrix multiplication
3. **Quantization** - Use Q4 models for better speed
4. **Batch processing** - Process multiple sequences together

## 🚀 Conclusion

The **iGPU-only approach is recommended** for transformer inference on AMD Phoenix APUs:
- Simpler implementation
- Nearly identical performance
- Better resource utilization
- Easier to debug and maintain

The NPU acceleration, while technically working, provides minimal real-world benefit for current transformer architectures where GEMM operations dominate the compute profile.