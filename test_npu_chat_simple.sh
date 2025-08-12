#!/bin/bash

echo "🦄 TESTING REAL NPU+iGPU CHAT!"
echo "=============================="
echo ""

# Simple one-shot test
./llama.cpp/build/bin/llama-cli \
    -m tinyllama-1.1b-q4_k_m.gguf \
    -p "Hello! What is NPU?" \
    --npu-attention \
    --gpu-layers 999 \
    -n 30 \
    --temp 0.7 \
    --no-warmup 2>&1 | tee /tmp/npu_test.log

echo ""
echo "📊 RESULTS:"
echo "-----------"

# Check NPU timing
if grep -q "NPU REAL attention computed" /tmp/npu_test.log; then
    npu_time=$(grep -oP "NPU REAL attention computed in \K[0-9]+" /tmp/npu_test.log | tail -1)
    echo "✅ NPU Processing: ${npu_time} μs"
else
    echo "❌ NPU processing not detected"
fi

# Check performance
tps=$(grep -oP "eval time.*\K[0-9.]+(?= tokens per second)" /tmp/npu_test.log | tail -1)
if [ ! -z "$tps" ]; then
    echo "✅ Performance: ${tps} tokens/second"
fi

# Check if we got actual response
if grep -q "NPU stands for" /tmp/npu_test.log || grep -q "Neural Processing Unit" /tmp/npu_test.log; then
    echo "✅ Chat Response: Generated successfully!"
else
    echo "⚠️  Chat Response: May have crashed"
fi

echo ""
echo "🎯 NPU+iGPU hybrid system status!"