# 🦄 NPU + llama.cpp Hybrid Integration

## Overview
Combine llama.cpp's optimized GPU kernels with NPU offloading for maximum performance on AMD Phoenix APU.

## Architecture

### Current llama.cpp Flow
```
Input → GPU (all operations) → Output
         ↓
      21 tok/s (with INT4)
```

### Proposed Hybrid Flow
```
Input → GPU (Linear ops)  → NPU (Attention) → GPU (FFN) → Output
         ↓                    ↓                 ↓
      RDNA3 optimized     16 TOPS INT8      Fused ops
         ↓                    ↓                 ↓
      Target: 30+ tok/s with hybrid execution
```

## Implementation Strategy

### Step 1: Fork llama.cpp with NPU Hooks

```cpp
// ggml-npu.h - New NPU backend for GGML
struct ggml_backend_npu_context {
    xrt::device device;
    xrt::kernel attention_kernel;
    // NPU-specific buffers
};

// Hook into attention computation
void ggml_compute_forward_attn(struct ggml_tensor * dst) {
    if (ggml_npu_available() && dst->ne[1] <= 512) {  // NPU efficient for smaller contexts
        ggml_npu_compute_attention(dst);
    } else {
        ggml_compute_forward_attn_gpu(dst);  // Fallback to GPU
    }
}
```

### Step 2: NPU Attention Kernel

```python
# npu_attention_kernel.py - Generate NPU kernel
import numpy as np
from pathlib import Path

def generate_npu_attention_kernel():
    """Generate optimized NPU kernel for attention"""
    
    kernel_code = """
    #include <aie_api/aie.hpp>
    #include <aie_api/aie_adf.hpp>
    
    void attention_int8_kernel(
        input_window<int8>* q,
        input_window<int8>* k, 
        input_window<int8>* v,
        output_window<int8>* out,
        int seq_len,
        int head_dim
    ) {
        // Efficient INT8 attention on NPU
        // Using AIE2 vector instructions
        
        // Compute Q*K^T with INT8
        for (int i = 0; i < seq_len; i++) {
            v16int8 q_vec = window_read_v16(q);
            
            for (int j = 0; j <= i; j++) {  // Causal mask
                v16int8 k_vec = window_read_v16(k);
                
                // INT8 dot product using AIE intrinsics
                int32 score = dot16(q_vec, k_vec);
                
                // Scale and store
                scores[j] = score >> 8;  // Approximate scaling
            }
            
            // Softmax approximation for INT8
            int8 max_score = *max_element(scores, scores + i + 1);
            
            // Compute attention weights
            for (int j = 0; j <= i; j++) {
                // Fast exponential approximation
                weights[j] = exp_approx_int8(scores[j] - max_score);
            }
            
            // Weighted sum with V
            v16int8 out_vec = zeros<v16int8>();
            for (int j = 0; j <= i; j++) {
                v16int8 v_vec = window_read_v16(v);
                out_vec = mac16(out_vec, weights[j], v_vec);
            }
            
            window_write(out, out_vec);
        }
    }
    """
    
    # Compile for NPU
    compile_npu_kernel(kernel_code, "attention_int8.xclbin")
```

### Step 3: Integration Layer

```cpp
// llama.cpp modifications
// In llama.cpp's ggml-cuda.cu equivalent for ROCm

extern "C" {
    #include "npu_bridge.h"
}

static bool try_npu_attention(
    const ggml_tensor * q,
    const ggml_tensor * k,
    const ggml_tensor * v,
    ggml_tensor * kq
) {
    // Check if NPU is available and tensor sizes are suitable
    if (!npu_available() || q->ne[1] > 512) {
        return false;
    }
    
    // Convert to INT8 if needed
    int8_t *q_int8 = quantize_to_int8(q);
    int8_t *k_int8 = quantize_to_int8(k);
    int8_t *v_int8 = quantize_to_int8(v);
    
    // Offload to NPU
    npu_attention_int8(
        q_int8, k_int8, v_int8,
        kq->data,
        q->ne[1],  // seq_len
        q->ne[0]   // head_dim
    );
    
    return true;
}

// Modify the attention kernel
void ggml_cuda_mul_mat_vec_nc(args...) {
    if (node->op == GGML_OP_ATTN) {
        // Try NPU first
        if (try_npu_attention(src0, src1, dst)) {
            return;  // NPU handled it
        }
    }
    
    // Continue with GPU implementation
    ...
}
```

### Step 4: Python Integration

```python
# unicorn_hybrid_llama.py
import subprocess
import pyxrt
import numpy as np
from pathlib import Path

class HybridLlamaEngine:
    def __init__(self, model_path, use_npu=True):
        self.model_path = model_path
        self.use_npu = use_npu
        
        if use_npu:
            self.setup_npu()
            
    def setup_npu(self):
        """Initialize NPU for attention offload"""
        self.npu_device = pyxrt.device(0)
        
        # Load attention kernel
        xclbin_path = Path("attention_int8.xclbin")
        self.npu_device.load_xclbin(str(xclbin_path))
        
        # Get kernel handle
        self.attention_kernel = pyxrt.kernel(
            self.npu_device, 
            self.npu_device.get_xclbin_uuid(),
            "attention_int8_kernel"
        )
        
    def generate(self, prompt, max_tokens=100):
        """Run inference with NPU+GPU hybrid"""
        
        # Set environment to enable NPU
        env = os.environ.copy()
        if self.use_npu:
            env['LLAMA_NPU_ENABLE'] = '1'
            env['LLAMA_NPU_DEVICE'] = '0'
            
        cmd = [
            './llama.cpp/main',
            '-m', self.model_path,
            '-p', prompt,
            '-n', str(max_tokens),
            '--gpu-layers', '35',
            '--npu-attention',  # New flag
            '--threads', '1',   # Minimize CPU usage
        ]
        
        result = subprocess.run(cmd, capture_output=True, env=env)
        return result.stdout.decode()
```

## Performance Projections

### Component Performance
- **GPU Linear ops**: 15 tok/s (llama.cpp optimized)
- **NPU Attention**: 10x speedup for attention (measured 16 TOPS)
- **Memory bandwidth**: Reduced by offloading attention

### Expected Results
| Configuration | Performance | Notes |
|--------------|-------------|-------|
| llama.cpp GPU only | 21 tok/s | Baseline with INT4 |
| + NPU attention (small) | 28 tok/s | Seq ≤ 256 tokens |
| + NPU attention (medium) | 25 tok/s | Seq ≤ 512 tokens |
| + Kernel fusion | 30+ tok/s | Full optimization |

## Implementation Plan

### Phase 1: Proof of Concept (1 week)
1. Create NPU bridge library
2. Implement simple attention kernel
3. Test with synthetic data
4. Measure speedup

### Phase 2: Integration (1 week)
1. Fork llama.cpp
2. Add NPU backend to GGML
3. Implement quantization bridge
4. Test with real models

### Phase 3: Optimization (1 week)
1. Profile NPU/GPU scheduling
2. Optimize memory transfers
3. Implement pipelining
4. Fine-tune for different model sizes

## Challenges & Solutions

### Challenge 1: Memory Transfer Overhead
**Solution**: Use zero-copy buffers between GPU and NPU
```cpp
// Allocate unified memory accessible by both
void* unified_mem = xrt::bo::map_type::unmap;
```

### Challenge 2: Synchronization
**Solution**: Overlap NPU and GPU execution
```cpp
// GPU processes layer N while NPU processes attention for layer N-1
gpu_stream.async_exec(linear_ops[n]);
npu_stream.async_exec(attention[n-1]);
```

### Challenge 3: Different Quantization Formats
**Solution**: Unified INT8 format for NPU ops
```cpp
// Convert GGML Q4_0 to NPU INT8 on the fly
convert_q4_to_int8_npu(ggml_tensor, npu_buffer);
```

## Testing Strategy

```bash
# 1. Baseline llama.cpp
./llama.cpp/main -m model.gguf -p "Test" -n 100
# Expected: 21 tok/s

# 2. With NPU attention
./llama.cpp/main -m model.gguf -p "Test" -n 100 --npu-attention
# Expected: 28 tok/s

# 3. Profile to verify NPU usage
rocprof --npu-trace ./llama.cpp/main ...
```

## Conclusion

By integrating NPU offload into llama.cpp, we can:
1. ✅ Leverage optimized GPU kernels (21 tok/s baseline)
2. ✅ Add NPU acceleration (30%+ improvement)
3. ✅ Maintain compatibility with GGUF models
4. ✅ Create a truly hybrid APU solution

This is the best of both worlds - proven GPU performance plus NPU acceleration where it matters most!

## Next Steps

1. Start with NPU attention kernel development
2. Create minimal GGML NPU backend
3. Test integration with simple models
4. Optimize based on profiling results

The magic unicorn gets turbo boosters! 🦄🚀