#!/bin/bash
# Quick NPU integration test script

echo "🦄 Quick NPU+iGPU Integration Test"
echo "=================================="

# Test with TinyLlama first (we know it loads)
MODEL="tinyllama-1.1b-q4_k_m.gguf"
LLAMA_CLI="llama.cpp/build/bin/llama-cli"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

if [ ! -f "$LLAMA_CLI" ]; then
    echo "❌ llama-cli not found: $LLAMA_CLI"
    echo "💡 Building llama.cpp..."
    cd llama.cpp
    cmake -B build -DGGML_VULKAN=ON -DGGML_NPU=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j8
    cd ..
fi

echo -e "\n🧪 Test 1: CPU Baseline"
echo "------------------------"
timeout 30 $LLAMA_CLI -m $MODEL -p "The NPU says" -n 10 --log-disable 2>&1 | grep -E "(tok/s|NPU|model loaded)"

echo -e "\n🧪 Test 2: Vulkan GPU"
echo "---------------------"
timeout 30 $LLAMA_CLI -m $MODEL -p "The GPU says" -n 10 --gpu-layers 999 --log-disable 2>&1 | grep -E "(tok/s|Vulkan|model loaded)"

echo -e "\n🧪 Test 3: NPU Attention (Real Kernel)"
echo "--------------------------------------"
timeout 30 $LLAMA_CLI -m $MODEL -p "The NPU kernel says" -n 10 --npu-attention --log-disable 2>&1 | grep -E "(tok/s|NPU|Direct Runtime|kernel)"

echo -e "\n🧪 Test 4: NPU+GPU Hybrid"
echo "-------------------------"
timeout 30 $LLAMA_CLI -m $MODEL -p "The hybrid says" -n 10 --npu-attention --gpu-layers 999 --log-disable 2>&1 | grep -E "(tok/s|NPU|Vulkan|hybrid)"

echo -e "\n✅ Quick test complete!"
echo "🚀 Ready for Gemma model testing"