# 🦄 Vulkan + NPU Hybrid: The Optimal Path

## Why Vulkan > ROCm for llama.cpp

### Performance Reality
- **Vulkan on RDNA3**: Often 20-50% faster than ROCm
- **Better optimization**: Vulkan compute shaders are highly optimized for gaming GPUs
- **Lower overhead**: Direct to metal, no HIP translation layer
- **Proven results**: Community reports 25-30 tok/s with Vulkan on similar hardware

### Vulkan Advantages
1. **Native AMD driver support** - No ROCm installation needed
2. **Cross-platform** - Works on Windows, Linux, even Android
3. **Better memory management** - Efficient for consumer GPUs
4. **Active development** - Vulkan backend regularly updated

## Architecture: Vulkan + NPU Hybrid

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Text Input    │     │  Vulkan (iGPU)  │     │   NPU (XDNA)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         ▼                       ▼                         ▼
    Tokenization          Linear Operations          Attention Heads
         │                  - QKV Project               - Q*K^T
         │                  - FFN Layers                - Softmax  
         │                  - Output Proj               - Attention
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                                 ▼
                         Combined Output
                          (30+ tok/s)
```

## Implementation Plan

### Step 1: Build llama.cpp with Vulkan

```bash
# Install Vulkan SDK
sudo apt update
sudo apt install vulkan-tools vulkan-sdk vulkan-validationlayers

# Verify Vulkan works with AMD GPU
vulkaninfo | grep -A5 "GPU id"

# Clone and build llama.cpp with Vulkan
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean
make LLAMA_VULKAN=1 -j8

# Test Vulkan performance
./main -m model.gguf -p "Hello" -n 100 --gpu-layers 999
```

### Step 2: Create NPU Attention Offload

```cpp
// vulkan_npu_bridge.cpp
#include <vulkan/vulkan.h>
#include "ggml-vulkan.h"
#include "npu_attention.h"

// Hook into Vulkan command buffer submission
void vk_npu_attention_offload(
    VkCommandBuffer cmd,
    const ggml_tensor* q,
    const ggml_tensor* k, 
    const ggml_tensor* v,
    ggml_tensor* output
) {
    // Check if suitable for NPU
    int seq_len = q->ne[1];
    int num_heads = q->ne[2];
    
    if (seq_len <= 512 && npu_available()) {
        // Flush Vulkan queue
        vkEndCommandBuffer(cmd);
        vkQueueSubmit(...);
        
        // Offload to NPU
        npu_attention_forward(
            q->data,
            k->data,
            v->data,
            output->data,
            seq_len,
            num_heads
        );
        
        // Sync and continue
        vkQueueWaitIdle(...);
        vkBeginCommandBuffer(cmd, ...);
    } else {
        // Continue with Vulkan
        vk_attention_compute(cmd, q, k, v, output);
    }
}
```

### Step 3: Modify GGML Vulkan Backend

```cpp
// In ggml-vulkan.cpp
static void ggml_vk_mul_mat_q(
    ggml_backend_vk_context* ctx,
    vk_context* subctx, 
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst
) {
    // Check if this is attention computation
    if (is_attention_pattern(src0, src1, dst)) {
        // Try NPU offload
        if (try_npu_attention_offload(src0, src1, dst)) {
            return; // NPU handled it
        }
    }
    
    // Continue with Vulkan implementation
    // ... existing Vulkan code ...
}
```

### Step 4: Optimal Workload Distribution

```python
# workload_optimizer.py
def get_optimal_distribution(model_size, context_length):
    """Determine optimal Vulkan/NPU split"""
    
    distributions = {
        # Model size -> (vulkan_layers, npu_attention_layers)
        "3B": {
            "short": (32, [0, 1, 2, 3]),      # First 4 layers to NPU
            "medium": (32, [0, 1]),            # First 2 layers to NPU
            "long": (32, []),                  # All on Vulkan
        },
        "7B": {
            "short": (35, [0, 1, 2, 3, 4]),   # First 5 layers to NPU
            "medium": (35, [0, 1, 2]),         # First 3 layers to NPU
            "long": (35, []),                  # All on Vulkan
        }
    }
    
    context_type = "short" if context_length <= 256 else \
                   "medium" if context_length <= 512 else "long"
    
    return distributions[model_size][context_type]
```

## Performance Benchmarks

### Expected Performance

| Configuration | Implementation | Expected Speed | Notes |
|--------------|----------------|----------------|-------|
| Vulkan only (FP32) | Baseline | 5-8 tok/s | No quantization |
| Vulkan only (INT4) | Optimized | 25-30 tok/s | GGUF Q4_K_M |
| Vulkan + NPU (INT4) | Hybrid | 32-38 tok/s | Attention offload |
| Vulkan + NPU + Opt | Ultimate | 40+ tok/s | With all optimizations |

### Why This Works Better

1. **Vulkan Efficiency**: Already achieves 90%+ GPU utilization
2. **NPU for Attention**: Attention is memory-bound, perfect for NPU
3. **No Overhead**: Vulkan has minimal CPU overhead
4. **Better Quantization**: Vulkan shaders handle INT4 efficiently

## Quick Test Script

```bash
#!/bin/bash
# test_vulkan_performance.sh

echo "🦄 Testing Vulkan + NPU Hybrid Performance"
echo "=========================================="

# Check Vulkan
if ! command -v vulkaninfo &> /dev/null; then
    echo "❌ Vulkan not found. Installing..."
    sudo apt install -y vulkan-tools vulkan-sdk
fi

# Build llama.cpp with Vulkan
if [ ! -f "llama.cpp/main" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make LLAMA_VULKAN=1 -j8
    cd ..
fi

# Download test model
if [ ! -f "tinyllama-1.1b-q4_k_m.gguf" ]; then
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

# Test configurations
echo -e "\n📊 Benchmarking Configurations:"
echo "--------------------------------"

# 1. CPU only baseline
echo -e "\n1. CPU Only:"
./llama.cpp/main -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Once upon a time" -n 50 --gpu-layers 0 2>&1 | grep "tok/s"

# 2. Vulkan GPU
echo -e "\n2. Vulkan GPU:"
./llama.cpp/main -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Once upon a time" -n 50 --gpu-layers 999 2>&1 | grep "tok/s"

# 3. Vulkan + NPU (when implemented)
echo -e "\n3. Vulkan + NPU (simulated):"
echo "   Expected: 35-40 tok/s with attention offload"
```

## Integration with Unicorn CLI

```python
# Update unicorn_cli.py
def load_engine(self, args):
    if args.backend == "vulkan":
        from unicorn_vulkan_backend import UnicornVulkanBackend
        self.engine = UnicornVulkanBackend(
            args.model,
            use_npu=args.device in ["npu", "hybrid"],
            npu_layers=args.npu_layers  # Which attention layers to offload
        )
```

## Why This is the Best Approach

1. **Proven Performance**: Vulkan already achieves our target on similar hardware
2. **Easier Integration**: Vulkan is more stable than ROCm on consumer GPUs
3. **NPU Synergy**: Attention is perfect workload for NPU (memory-bound)
4. **Cross-Platform**: Works on Windows too (important for wider adoption)
5. **Active Development**: Vulkan backend actively maintained

## Next Steps

1. **Today**: Build and test llama.cpp with Vulkan
2. **This Week**: Verify 25+ tok/s with Vulkan alone
3. **Next Week**: Add NPU attention offload
4. **Final**: Optimize workload distribution

The magic unicorn flies fastest with Vulkan wings and NPU turbo! 🦄✨