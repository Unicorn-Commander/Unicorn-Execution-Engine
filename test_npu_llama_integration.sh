#!/bin/bash
# Test NPU integration with llama.cpp

echo "🦄 Testing NPU + Vulkan llama.cpp Integration"
echo "============================================"

# Set environment
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"

# Check if model exists
MODEL="tinyllama-1.1b-q4_k_m.gguf"
if [ ! -f "$MODEL" ]; then
    echo "📥 Downloading test model..."
    wget -q --show-progress \
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
        -O "$MODEL"
fi

echo ""
echo "🚀 Running inference test..."
echo "Prompt: 'The future of AI acceleration is'"
echo ""

# Run with Vulkan + NPU
time ./llama.cpp/build/bin/llama-cli \
    -m "$MODEL" \
    -p "The future of AI acceleration is" \
    -n 50 \
    --gpu-layers 999 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    2>&1 | tee npu_test_output.txt

echo ""
echo "📊 Performance Analysis:"
grep -E "(tok/s|eval time|load time)" npu_test_output.txt || echo "No performance data found"

echo ""
echo "🔍 NPU Status Check:"
if grep -q "NPU" npu_test_output.txt; then
    echo "✅ NPU backend detected in output"
else
    echo "⚠️  NPU backend not visible in output"
fi

echo ""
echo "📄 Full output saved to: npu_test_output.txt"
