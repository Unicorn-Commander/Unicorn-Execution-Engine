# Quick Fix: Skip LoRA for NPU Testing

## The Issue
NPU attention works perfectly but fails when the graph builder tries to apply LoRA operations to the NPU output tensor.

## Quick Solution
Modify `llama-graph.cpp` to skip LoRA operations when NPU attention is used.

## File to Modify
`/home/ucadmin/Development/Unicorn-Execution-Engine/llama.cpp/src/llama-graph.cpp`

## Changes Needed

### Around line 1330-1332:
```cpp
// BEFORE:
if (wo) {
    cur = build_lora_mm(wo, cur);
}

// AFTER:
if (wo && !cparams.use_npu_attention) {
    cur = build_lora_mm(wo, cur);
}
```

### Similar changes needed at:
- Line 1419-1420 (build_attn function)
- Line 1485-1486 (another build_attn variant)
- Line 1539-1540 (yet another variant)

## Alternative: Conditional NPU Path
Instead of modifying existing code, create a separate path:

```cpp
if (cparams.use_npu_attention) {
    // NPU path - skip LoRA
    if (wo) {
        cur = ggml_mul_mat(ctx0, wo, cur);
    }
} else {
    // Regular path with LoRA
    if (wo) {
        cur = build_lora_mm(wo, cur);
    }
}
```

## Testing
After making these changes:
1. Rebuild: `cmake --build build --config Release -j8`
2. Test: `./build/bin/llama-cli -m model.gguf -p "Hello" -n 10 --npu-attention`

## Expected Result
- NPU attention executes successfully
- No LoRA multiplication errors
- Model generates tokens using NPU acceleration

This is a temporary fix to prove NPU functionality. A proper solution would handle LoRA tensors correctly with NPU output.