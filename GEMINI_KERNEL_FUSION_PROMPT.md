# 🚀 Kernel Fusion Task for Gemini

## Context
I need you to create fused GPU kernels for transformer inference that eliminate kernel launch overhead. We've proven the AMD Radeon Phoenix iGPU (gfx1103) can achieve 897 GFLOPS on GEMM, but the full transformer pipeline is slow due to many small kernel launches.

## Current Situation
- **Location**: `/home/ucadmin/Development/Unicorn-Execution-Engine/`
- **GPU**: AMD Radeon Phoenix (gfx1103) with 6 compute units
- **Current Performance**: 0.3 TPS (need 10+ TPS)
- **Bottleneck**: Kernel launch overhead for small operations

## Files to Review
```bash
# Current optimized kernels (897 GFLOPS GEMM)
/home/ucadmin/Development/Unicorn-Execution-Engine/igpu_optimized_kernels.py

# Analysis showing why it's slow
/home/ucadmin/Development/Unicorn-Execution-Engine/NPU_IGPU_ANALYSIS.md

# Model configurations
/home/ucadmin/Development/Unicorn-Execution-Engine/igpu_final_pipeline.py
```

## Task: Create Fused Transformer Kernel

### Goal
Create a single OpenCL kernel that processes an entire transformer layer without returning to host code. This eliminates kernel launch overhead.

### Specific Requirements

1. **Fused Operations** - Combine in ONE kernel:
   - QKV projection (GEMM)
   - Attention computation (Q @ K^T)
   - Softmax
   - Value aggregation (Attn @ V)
   - Output projection (GEMM)
   - Layer normalization
   - MLP: Gate and Up projections (GEMM)
   - GELU activation
   - Down projection (GEMM)
   - Residual connections

2. **Memory Management**:
   - Keep all intermediate results in GPU memory
   - Use Local Data Share (LDS) for tiles
   - Minimize global memory access

3. **Target Specifications**:
   - Gemma 3 4B: hidden_size=2560, num_heads=20, ff_dim=10240
   - Gemma 3 27B: hidden_size=4608, num_heads=32, ff_dim=18432
   - Support sequence lengths: 128, 512, 2048

### Technical Approach

```opencl
__kernel void transformer_layer_fused(
    __global const float* input,        // [batch, seq_len, hidden_size]
    __global const float* weights,      // All layer weights concatenated
    __global float* output,            // [batch, seq_len, hidden_size]
    __local float* lds,                // Shared memory
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_heads,
    const int ff_dim
) {
    // 1. Load input tile to LDS
    // 2. QKV projection in-place
    // 3. Attention computation with tiling
    // 4. Output projection
    // 5. Layer norm in LDS
    // 6. MLP operations
    // 7. Write final output
    
    // NO RETURN TO HOST UNTIL COMPLETE LAYER IS DONE
}
```

## Checklist for Implementation

- [ ] **Analyze memory requirements**
  - Calculate LDS usage for different tile sizes
  - Ensure fits in 64KB LDS limit
  - Plan weight layout in global memory

- [ ] **Implement tiled QKV projection**
  - Fuse three GEMMs into one operation
  - Output directly to attention format

- [ ] **Implement Flash Attention algorithm**
  - Process attention in tiles to fit in LDS
  - Fuse softmax into attention computation
  - Handle causal masking efficiently

- [ ] **Implement fused MLP**
  - Combine gate/up projections
  - Fuse GELU activation
  - Pipeline with down projection

- [ ] **Add layer normalization**
  - Implement in LDS for efficiency
  - Fuse with residual connections

- [ ] **Optimize memory access patterns**
  - Coalesce global memory reads/writes
  - Use vector loads (float4) where possible
  - Minimize bank conflicts in LDS

- [ ] **Test and benchmark**
  - Verify numerical accuracy
  - Measure GFLOPS utilization
  - Profile kernel execution time

## Expected Outcome

A single kernel call that processes entire transformer layer:
```python
# Instead of:
# - 10+ kernel launches
# - Multiple CPU-GPU synchronizations
# - Memory transfers between operations

# Just one call:
fused_kernel(input, all_weights, output, ...)
queue.finish()  # Only one sync per layer
```

## Performance Target
- **Current**: 1166ms per layer (28 kernels launched)
- **Target**: <50ms per layer (1 kernel launched)
- **Expected speedup**: 20x+

## Additional Resources

AMD ROCm documentation for RDNA2 optimizations:
- Use wavefront-level primitives
- Optimize for 64-thread wavefronts
- Leverage matrix core instructions if available

## Complexity Assessment

**Difficulty: High** (8/10)
- Requires deep understanding of GPU architecture
- Complex memory management
- Numerical precision concerns
- Large kernel (1000+ lines of code)

**Time estimate**: 2-3 days for experienced GPU programmer

## Alternative: NPU Kernel Development

If kernel fusion proves too complex, consider implementing NPU kernels instead:

```bash
# NPU kernel template location
/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_source/

# Use Vitis AI to compile for Phoenix NPU
vitis_ai_compiler --target phoenix_npu ...
```

NPU is specifically designed for transformers and might be easier to get working efficiently.

---

**Note to Gemini**: This is a complex task requiring expert-level GPU programming. Focus on correctness first, then optimize. The key insight is that we need to eliminate the overhead of launching many small kernels by fusing everything into one large, efficient kernel.