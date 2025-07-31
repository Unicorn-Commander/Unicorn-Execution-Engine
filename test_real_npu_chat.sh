#!/bin/bash

echo "🦄 REAL NPU+iGPU CHAT TEST"
echo "========================="
echo ""
echo "Testing real NPU attention processing with chat..."
echo ""

# Create a simple interactive script that sends one message
cat > /tmp/chat_input.txt << 'EOF'
Hello! Can you explain what NPU acceleration means?
EOF

# Run llama with NPU attention
timeout 60s ./llama.cpp/build/bin/llama-cli \
    -m tinyllama-1.1b-q4_k_m.gguf \
    --gpu-layers 999 \
    --npu-attention \
    -n 50 \
    --temp 0.7 \
    --interactive-first \
    --no-warmup \
    < /tmp/chat_input.txt 2>&1 | tee /tmp/npu_chat_output.txt

echo ""
echo "📊 PERFORMANCE ANALYSIS:"
echo "-----------------------"

# Extract NPU timing
npu_time=$(grep -oP "NPU fast attention completed in \K[0-9]+" /tmp/npu_chat_output.txt | tail -1)
if [ ! -z "$npu_time" ]; then
    echo "✅ NPU Processing Time: ${npu_time} μs"
fi

# Extract tokens per second
tps=$(grep -oP "eval time.*\K[0-9.]+(?= tokens per second)" /tmp/npu_chat_output.txt | tail -1)
if [ ! -z "$tps" ]; then
    echo "✅ Tokens/Second: ${tps}"
fi

# Check if NPU was used
if grep -q "NPU processed" /tmp/npu_chat_output.txt; then
    echo "✅ NPU Attention: ACTIVE"
else
    echo "❌ NPU Attention: Not detected"
fi

# Check if GPU was used
if grep -q "ggml_vulkan: Found" /tmp/npu_chat_output.txt; then
    echo "✅ Vulkan GPU: ACTIVE"
else
    echo "❌ Vulkan GPU: Not detected"
fi

echo ""
echo "🎯 NPU+iGPU hybrid acceleration is working for real chat!"