#!/bin/bash
# Test llama.cpp with Vulkan

echo "🦄 Vulkan + llama.cpp Performance Test"
echo "======================================"

# Check if llama-cli exists
if [ ! -f "llama.cpp/build/bin/llama-cli" ]; then
    echo "❌ llama-cli not found. Please build first."
    exit 1
fi

# Test Vulkan capability
echo -e "\n📊 Testing Vulkan backend..."
echo "Note: This will show available backends even without a model"

# Run with --help to check Vulkan support
./llama.cpp/build/bin/llama-cli --help 2>&1 | grep -i vulkan && echo "✅ Vulkan backend available!" || echo "❌ Vulkan backend not detected"

# Check system info
echo -e "\n🖥️ System Information:"
./llama.cpp/build/bin/llama-cli --version 2>&1 | head -5

# Create a benchmark script for when model is available
cat > run_benchmark.sh << 'EOF'
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
EOF

chmod +x run_benchmark.sh

echo -e "\n📋 Next Steps:"
echo "1. Download a GGUF model:"
echo "   - Visit https://huggingface.co/models?search=gguf"
echo "   - Look for Q4_K_M quantized models (good balance)"
echo "   - Download with: wget <model_url>"
echo ""
echo "2. Run benchmark:"
echo "   ./run_benchmark.sh <model.gguf>"
echo ""
echo "3. Expected performance:"
echo "   - CPU: 1-5 tok/s"
echo "   - Vulkan: 25-35 tok/s"
echo "   - Vulkan+NPU: 35-45 tok/s (after integration)"

# Create NPU integration stub
echo -e "\n🔧 Creating NPU integration framework..."
mkdir -p llama-npu-integration