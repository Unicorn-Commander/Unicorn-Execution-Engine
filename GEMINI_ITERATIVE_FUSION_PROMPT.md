# 🚀 Iterative Kernel Fusion Task for AMD iGPU

## Context and Goal
We need to optimize transformer inference on AMD Radeon Phoenix iGPU (gfx1103) by fusing multiple kernel operations. Currently achieving only 897 GFLOPS out of 8.294 TFLOPS theoretical peak (10.8% utilization). The bottleneck is kernel launch overhead from 28+ separate operations per transformer layer.

**Current Performance**: 0.3 TPS (Gemma 3 4B), 0.1 TPS (Gemma 3 27B)
**Target**: 10+ TPS (4B), 5+ TPS (27B)
**Strategy**: Iterative kernel fusion to reduce kernel launches from 28 to eventually 1-2

## Environment Details
- **Working Directory**: `/home/ucadmin/Development/Unicorn-Execution-Engine/`
- **GPU**: AMD Radeon Phoenix (gfx1103)
  - Compute Units: 6
  - Wavefront Size: 64
  - Local Memory: 64 KB
  - Peak Performance: 8.294 TFLOPS
- **Python**: 3.13
- **OpenCL**: 2.1

## Key Files to Review

```bash
# Current kernel implementations (achieving 897 GFLOPS on GEMM)
/home/ucadmin/Development/Unicorn-Execution-Engine/igpu_optimized_kernels.py

# Analysis showing why current approach is slow
/home/ucadmin/Development/Unicorn-Execution-Engine/NPU_IGPU_ANALYSIS.md

# Current pipeline using many small kernels
/home/ucadmin/Development/Unicorn-Execution-Engine/igpu_final_pipeline.py

# Iterative fusion plan and expected performance
/home/ucadmin/Development/Unicorn-Execution-Engine/ITERATIVE_KERNEL_FUSION_PLAN.md

# Phase 1 starter kernels (QKV fusion example)
/home/ucadmin/Development/Unicorn-Execution-Engine/PHASE1_QKV_FUSION_STARTER.cl

# Model configurations
# Gemma 3 4B: hidden_size=2560, num_heads=20, head_dim=128, ff_dim=10240
# Gemma 3 27B: hidden_size=4608, num_heads=32, head_dim=144, ff_dim=18432
```

## Phase 1 Tasks (Start Here)

### Task 1.1: Complete QKV Projection Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/qkv_fused.cl`

Building on the starter code in `PHASE1_QKV_FUSION_STARTER.cl`, create a production-ready fused QKV kernel that:
- Combines 3 separate GEMM operations into one
- Uses tiled computation with Local Data Share (LDS)
- Supports both 4B and 27B model dimensions
- Handles arbitrary batch sizes and sequence lengths

**Test with**:
```python
# Create test file: /home/ucadmin/Development/Unicorn-Execution-Engine/test_qkv_fusion.py
import pyopencl as cl
import numpy as np
import time

# Test the fused kernel vs separate operations
# Measure speedup: should be ~2x faster
```

### Task 1.2: Attention Score + Softmax Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/attention_softmax_fused.cl`

Fuse attention score computation with softmax:
- Compute Q @ K^T and immediately apply softmax
- Implement causal masking efficiently
- Use stable softmax (subtract max before exp)
- Eliminate intermediate storage of raw scores

### Task 1.3: MLP Block Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_fused.cl`

Fuse the entire MLP block:
```
Input -> Gate projection -> GELU activation -> Up projection -> Down projection -> Output
```
- Fuse gate and up projections (can share input reads)
- Apply GELU in-place
- Pipeline with down projection

### Task 1.4: Integration and Testing
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/phase1_fused_pipeline.py`

Integrate the Phase 1 kernels into a working pipeline:
```python
class Phase1FusedPipeline:
    def __init__(self, model_type="4b"):
        # Load Phase 1 fused kernels
        self.load_fused_kernels()
    
    def forward_layer(self, hidden_states):
        # Use fused kernels instead of separate ops
        # Should use ~15 kernel launches instead of 28
```

## Phase 2 Tasks (After Phase 1 Works)

### Task 2.1: Complete Attention Block Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase2/attention_block_fused.cl`

Combine all Phase 1 attention kernels:
- QKV projection (from 1.1)
- Attention scores + softmax (from 1.2)
- Attention @ V multiplication
- Output projection
- Residual connection

### Task 2.2: Complete MLP Block Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase2/mlp_block_fused.cl`

Extend Phase 1 MLP fusion to include:
- Layer normalization before MLP
- Residual connection after MLP
- All in one kernel

## Phase 3 Tasks (Ultimate Goal)

### Task 3.1: Full Transformer Layer Fusion
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase3/transformer_layer_fused.cl`

Combine everything into one massive kernel (if memory permits):
- Complete attention block
- First layer norm
- Complete MLP block  
- Second layer norm
- Both residual connections

## Testing and Benchmarking

### Create Benchmark Suite
**File**: Create `/home/ucadmin/Development/Unicorn-Execution-Engine/benchmark_fusion.py`

```python
import time
import numpy as np
import pyopencl as cl

def benchmark_phase(phase_name, pipeline, seq_lengths=[128, 512, 2048]):
    """Benchmark a fusion phase"""
    results = {}
    
    for seq_len in seq_lengths:
        # Create test input
        input_data = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            output = pipeline.forward(input_data)
        
        # Benchmark
        start = time.time()
        iterations = 50
        for _ in range(iterations):
            output = pipeline.forward(input_data)
        elapsed = time.time() - start
        
        # Calculate metrics
        ms_per_layer = (elapsed / iterations) * 1000
        tps = estimate_tps(ms_per_layer, num_layers)
        
        results[seq_len] = {
            'ms_per_layer': ms_per_layer,
            'tps': tps,
            'speedup': tps / baseline_tps
        }
    
    return results
```

## Expected Performance Progression

| Phase | Kernel Launches | Expected GFLOPS | Expected TPS (4B) | Files to Create |
|-------|----------------|-----------------|-------------------|-----------------|
| Baseline | 28 | 897 | 0.3 | - |
| Phase 1 | 15 | 2,000 | 1-2 | 4 kernel files + test |
| Phase 2 | 4-6 | 4,000 | 3-5 | 2 kernel files |
| Phase 3 | 1-2 | 6,000 | 8-15 | 1 kernel file |

## Build and Compilation

```bash
# OpenCL kernels compile automatically when loaded
# Use these build options for optimization:
build_options = """
-cl-std=CL2.0
-cl-fast-relaxed-math
-cl-mad-enable
-cl-denorms-are-zero
-cl-no-signed-zeros
-cl-finite-math-only
"""
```

## Debugging Tips

1. **Start with correctness**: Compare fused kernel output with original
2. **Profile each kernel**: Use `cl.enqueue_marker` and event timing
3. **Check occupancy**: Ensure full GPU utilization
4. **Monitor memory usage**: Stay within 64KB LDS limit

## AMD-Specific Optimizations

- Use 64-thread wavefronts (natural for RDNA2)
- Align memory accesses to 128-byte boundaries
- Use `__attribute__((reqd_work_group_size(X, Y, Z)))`
- Leverage `ds_permute` instructions for shuffle operations

## Success Criteria

### Phase 1 Success:
- [ ] All Phase 1 kernels compile and run
- [ ] Numerical accuracy within 1e-5 of original
- [ ] 2x speedup achieved (0.3 → 0.6+ TPS)
- [ ] Kernel launches reduced from 28 to ~15

### Phase 2 Success:
- [ ] Phase 2 kernels integrate Phase 1 work
- [ ] 5x total speedup achieved (0.3 → 1.5+ TPS)
- [ ] Kernel launches reduced to 4-6

### Phase 3 Success:
- [ ] Single kernel processes entire layer
- [ ] 10x+ speedup achieved (0.3 → 3+ TPS)
- [ ] Approaching 50%+ of theoretical peak FLOPS

## Start with Phase 1, Task 1.1

Begin by completing the QKV fusion kernel. This is the easiest fusion and will immediately show if the approach is working. Once you see the 2x speedup from eliminating 2 kernel launches, proceed to the next fusion.

Remember: Each successful fusion makes the next one easier as you learn the patterns and techniques!

---

**Note**: This is an iterative process. Don't try to optimize everything at once. Get each phase working and benchmarked before moving to the next. The key insight is that even Phase 1 alone would make the system 2x faster, which is already a significant improvement.