#!/bin/bash
# Benchmark script for llama.cpp with Vulkan

MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf>"
    exit 1
fi

echo "🚀 Benchmarking $MODEL with Vulkan..."
echo "====================================="

# Test configurations
PROMPTS=(
    "The key to artificial intelligence is"
    "Once upon a time in a land far away"
    "The future of computing lies in"
)

# CPU baseline
echo -e "\n1️⃣ CPU Baseline (no GPU):"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "${PROMPTS[0]}" \
    -n 50 \
    --no-display-prompt \
    --gpu-layers 0 \
    2>&1 | grep -E "(tok/s|tokens per second)"

# Vulkan GPU test
echo -e "\n2️⃣ Vulkan GPU (all layers):"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "${PROMPTS[0]}" \
    -n 50 \
    --no-display-prompt \
    --gpu-layers 999 \
    2>&1 | grep -E "(tok/s|tokens per second)"

# Longer context test
echo -e "\n3️⃣ Vulkan GPU (longer context):"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "${PROMPTS[1]}" \
    -n 200 \
    --no-display-prompt \
    --gpu-layers 999 \
    2>&1 | grep -E "(tok/s|tokens per second)"

# Batch test
echo -e "\n4️⃣ Vulkan GPU (batch mode):"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "${PROMPTS[2]}" \
    -n 100 \
    --no-display-prompt \
    --gpu-layers 999 \
    --batch-size 512 \
    2>&1 | grep -E "(tok/s|tokens per second)"
