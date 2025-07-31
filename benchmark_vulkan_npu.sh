#!/bin/bash
# Benchmark Vulkan (and future NPU) performance

MODEL=${1:-tinyllama-1.1b-q4_k_m.gguf}
PROMPT="The future of AI acceleration is"

echo "🚀 Benchmarking: $MODEL"
echo "================================"

# CPU baseline
echo -e "\n1️⃣ CPU Performance:"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "$PROMPT" -n 50 \
    --no-display-prompt \
    --gpu-layers 0 \
    2>&1 | grep -E "(tok/s|eval time)"

# Vulkan GPU
echo -e "\n2️⃣ Vulkan GPU Performance:"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "$PROMPT" -n 50 \
    --no-display-prompt \
    --gpu-layers 999 \
    2>&1 | grep -E "(tok/s|eval time|Vulkan)"

# Future: Vulkan + NPU
echo -e "\n3️⃣ Vulkan + NPU (Coming Soon):"
echo "When NPU backend is integrated:"
echo "  - Vulkan will handle linear operations"
echo "  - NPU will handle attention (INT8)"
echo "  - Expected: 25-35% performance boost"

# Show system info
echo -e "\n📊 System Info:"
echo "  GPU: $(vulkaninfo 2>/dev/null | grep deviceName | head -1 | cut -d'=' -f2 | xargs)"
echo "  NPU: AMD Phoenix NPU (16 TOPS)"
echo "  Driver: $(lsmod | grep amdxdna | head -1)"
