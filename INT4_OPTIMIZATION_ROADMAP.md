# 🦄 INT4 Optimization Roadmap

## Executive Summary

Our quantization analysis discovered that **INT4 quantization is the key to reaching the 21 tok/s ollama baseline**. This roadmap details the complete implementation path combining INT4 quantization with existing infrastructure.

### Key Discovery
- **Current performance**: 0.366 tok/s (FP32 baseline)
- **Target performance**: 21 tok/s (57.4x improvement needed)
- **INT4 theoretical speedup**: 7.5x (reducing gap from 57x to 7.5x)
- **Remaining optimizations needed**: 7.5x (achievable with kernels + NPU)

## Phase 1: INT4 WMMA Implementation (In Progress with Gemini)

### 1.1 HIP/ROCm WMMA Kernel Development
**Status**: Gemini implementing
**Target**: 7-8x speedup over FP32

Key components:
- `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` intrinsic
- 1024 FLOPS/clock/CU for INT4 operations
- 16x16 tile-based matrix multiplication
- Wave32 cooperative execution

### 1.2 Validation Benchmarks
**Status**: Pending Gemini's implementation
**Target**: Verify 2.8 tok/s achieved

Tests to run:
```bash
# Quick validation
python3.13 unicorn_benchmark_suite.py --quick --device igpu --quantization int4

# Full validation
python3.13 test_hip_int4_performance.py
```

Expected results:
- Layer time: ~3ms (down from 125ms)
- Full model: ~126ms (42 layers)
- Speed: 7.9+ tokens/sec

## Phase 2: Integration with Existing Pipeline

### 2.1 Merge INT4 WMMA with Hybrid Pipeline
Integrate Gemini's HIP WMMA kernels into `optimized_hybrid_pipeline.py`:

```python
class OptimizedHybridEngineINT4(OptimizedHybridEngine):
    def __init__(self):
        super().__init__()
        self.use_int4_wmma = self._check_wmma_support()
        if self.use_int4_wmma:
            self.hip_int4_wmma = load_hip_int4_wmma()
            
    def forward_layer_optimized_int4(self, x, weights):
        if self.use_int4_wmma:
            # Use INT4 WMMA for linear ops
            return self.hip_int4_wmma.forward(x, weights)
        else:
            # Fallback to OpenCL
            return super().forward_layer_optimized(x, weights)
```

### 2.2 Model Loading with INT4 Quantization
Extend `magic_unicorn_model_loader.py` to support INT4 WMMA format:

```python
def prepare_int4_wmma_weights(self, weights):
    """Prepare weights in INT4 WMMA-friendly format"""
    # Pack weights into 16x16 tiles
    # Ensure proper memory layout for WMMA
    # Apply INT4 quantization with proper scaling
```

## Phase 3: Memory Optimization

### 3.1 Zero-Copy INT4 Buffers
- Allocate INT4 buffers directly on iGPU
- Avoid CPU↔GPU transfers
- Use pinned memory for model weights

### 3.2 Activation Caching
- Keep intermediate activations in INT4 format
- Only dequantize for attention computation
- Reduces memory bandwidth by 8x

## Phase 4: NPU Integration for Attention

### 4.1 NPU Attention Kernel Development
While linear ops use INT4 WMMA on iGPU, attention can leverage NPU:

```python
def hybrid_attention_int4(q, k, v):
    """NPU-accelerated attention with INT4 inputs"""
    # Dequantize only Q,K for attention scores
    # Keep V in INT4 format
    # Use NPU's 16 TOPS for attention computation
```

### 4.2 Pipeline Optimization
- Overlap NPU attention with iGPU linear ops
- Pipeline execution across layers
- Hide NPU kernel launch latency

## Phase 5: Production Optimization

### 5.1 Dynamic Quantization
- Quantize activations on-the-fly
- Per-channel quantization for weights
- Adaptive scaling based on input range

### 5.2 Kernel Fusion
Fuse operations to reduce kernel launch overhead:
- QKV projection fusion
- Activation + projection fusion
- LayerNorm + next operation fusion

### 5.3 Multi-Stream Execution
- Use multiple HIP streams
- Overlap compute and memory transfers
- Pipeline multiple tokens

## Performance Projections

### With INT4 WMMA Only
- **Speedup**: 7.5x
- **Performance**: 2.8 tok/s
- **Gap to target**: 7.5x

### With Full Optimization Stack
| Optimization | Speedup | Cumulative | Performance |
|-------------|---------|------------|-------------|
| INT4 WMMA | 7.5x | 7.5x | 2.8 tok/s |
| NPU Attention | 2x | 15x | 5.6 tok/s |
| Kernel Fusion | 1.5x | 22.5x | 8.4 tok/s |
| Memory Opt | 1.5x | 33.8x | 12.6 tok/s |
| Multi-Stream | 1.7x | 57.4x | 21 tok/s ✓ |

## Implementation Timeline

### Week 1: INT4 WMMA Validation
- Complete Gemini's HIP kernel implementation
- Validate 7.5x speedup achieved
- Integration with existing pipeline

### Week 2: Memory & NPU Integration
- Implement zero-copy INT4 buffers
- Develop NPU attention kernel
- Test hybrid INT4 pipeline

### Week 3: Production Optimization
- Implement kernel fusion
- Add multi-stream execution
- Final performance validation

## Success Metrics

1. **Immediate (INT4 WMMA)**: 
   - ✓ 2.8+ tok/s achieved
   - ✓ Validated on real hardware
   - ✓ Integrated with CLI

2. **Short-term (Full Stack)**:
   - ✓ 21+ tok/s achieved
   - ✓ Ollama baseline matched
   - ✓ Production-ready

3. **Long-term**:
   - ✓ 50+ tok/s stretch goal
   - ✓ Multi-model support
   - ✓ Streaming inference

## Risk Mitigation

1. **INT4 Quality**: Implement mixed-precision fallback for sensitive layers
2. **Hardware Compatibility**: Maintain OpenCL fallback path
3. **Memory Pressure**: Implement activation checkpointing if needed

## Conclusion

INT4 quantization with RDNA3 WMMA is the critical breakthrough that makes the 21 tok/s target achievable. Combined with NPU attention and optimization techniques, we can match and exceed the ollama baseline on consumer AMD hardware.

The path is clear: **INT4 WMMA → NPU Integration → Optimization → Success** 🦄⚡