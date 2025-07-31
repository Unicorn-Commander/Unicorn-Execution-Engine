# Fix GGML Graph Building for NPU Attention

## Context
We're integrating NPU (Neural Processing Unit) acceleration into llama.cpp for the AMD Phoenix NPU. The NPU attention function is being called successfully, but we're hitting GGML graph building assertions.

## Current Status
- ✅ NPU hardware initializes correctly (Phoenix NPU, AIE 1.1)
- ✅ NPU kernels load successfully from `/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_inference/`
- ✅ The `--npu-attention` flag works and triggers NPU code path
- ❌ GGML graph building fails with: `GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor) failed`

## File to Fix
`/home/ucadmin/Development/Unicorn-Execution-Engine/llama.cpp/npu_stub.cpp`

## Current Implementation
The NPU attention function creates a new tensor but GGML doesn't know how to handle it in the graph:

```cpp
struct ggml_tensor * result = ggml_new_tensor_4d(ctx, q->type, 
                                                 head_dim,    // ne[0]
                                                 seq_len,     // ne[1]  
                                                 num_heads,   // ne[2]
                                                 batch_size); // ne[3]

if (result) {
    // Mark this tensor as the result of NPU attention operation
    // This will be computed during graph execution
    result->op = GGML_OP_NONE; // Placeholder op
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    
    printf("✅ NPU ATTENTION COMPLETE (tensor created for graph building)\n");
    printf("✅ NPU ATTENTION SUCCESS!\n");
    return result;
}
```

## What Needs to be Done

1. **Fix the tensor operation type**: Instead of `GGML_OP_NONE`, we need to either:
   - Use `GGML_OP_FLASH_ATTN_EXT` and hook into the existing flash attention infrastructure
   - Create a custom `GGML_OP_NPU_FLASH_ATTN` operation
   - Call the regular `ggml_flash_attn_ext()` but mark it for NPU execution

2. **Ensure proper graph integration**: The tensor must be properly added to the computation graph.

3. **Hook the compute function**: The `compute_npu_flash_attn_ext()` function at line 170 needs to be called during graph execution.

## Testing
After fixing, test with:
```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine
./llama.cpp/build/bin/llama-cli -m gemma-3n-E4B-it-Q8_0.gguf -p "The magic unicorn represents" -n 10 --npu-attention
```

## Expected Outcome
The model should run without crashes and show the NPU kernel loading messages:
```
🧠 NPU ATTENTION CALLED - Implementing REAL attention computation!
📋 Selected Gemma3n NPU kernel
🎯 Loading NPU kernel: ../npu_kernels_inference/gemma3n/attention_s512.npu
```

## Additional Context
- We have 15 compiled NPU kernels for different models and sequence lengths
- The NPU uses Direct IOCTL interface to `/dev/accel/accel0`
- XRT library integration is planned but not yet implemented
- The goal is to achieve 200x+ speedup on attention operations

Please fix the GGML graph building issue so the NPU attention can execute properly.