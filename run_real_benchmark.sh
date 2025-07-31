#!/bin/bash
# Real performance benchmark for NPU acceleration

echo "🚀 Unicorn Execution Engine - Real Performance Benchmark"
echo "======================================================"
echo ""

# Set up environment
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH

# Find the model
MODEL=""
if [ -f "gemma-2b-it-q4_k_m.gguf" ]; then
    MODEL="gemma-2b-it-q4_k_m.gguf"
elif [ -f "tinyllama-1.1b-q4_k_m.gguf" ]; then
    MODEL="tinyllama-1.1b-q4_k_m.gguf"
elif [ -f "gemma-3n-E4B-it-Q8_0.gguf" ]; then
    MODEL="gemma-3n-E4B-it-Q8_0.gguf"
else
    echo "❌ No model found. Downloading TinyLlama for testing..."
    wget -q https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O tinyllama-1.1b-q4_k_m.gguf
    MODEL="tinyllama-1.1b-q4_k_m.gguf"
fi

echo "📦 Using model: $MODEL"

# Find llama-cli binary
LLAMA_CLI=""
PATHS=(
    "/home/ucadmin/Development/Unicorn-Execution-Engine/llama.cpp/build/bin/llama-cli"
    "./llama.cpp/build/bin/llama-cli"
    "./build/bin/llama-cli"
    "./llama-cli"
)

for path in "${PATHS[@]}"; do
    if [ -f "$path" ] && [ -x "$path" ]; then
        LLAMA_CLI="$path"
        break
    fi
done

if [ -z "$LLAMA_CLI" ]; then
    echo "❌ llama-cli not found. Looking for any working binary..."
    
    # Try to find any llama binary
    LLAMA_CLI=$(find . -name "llama-cli" -type f -executable 2>/dev/null | head -1)
    
    if [ -z "$LLAMA_CLI" ]; then
        # Try the llama-server or other variants
        LLAMA_CLI=$(find . -name "llama-*" -type f -executable | grep -E "(llama-cli|llama-server|llama-simple)" | head -1)
    fi
    
    if [ -z "$LLAMA_CLI" ]; then
        echo "❌ No llama binary found. Please build llama.cpp first."
        exit 1
    fi
fi

echo "✅ Found binary: $LLAMA_CLI"
echo ""

# Test prompt
PROMPT="Once upon a time in a magical forest, there lived a wise old owl who loved to tell stories to all the woodland creatures. One day, the owl began a new tale:"

# Number of tokens to generate
N_TOKENS=100

echo "📝 Test configuration:"
echo "   Prompt: \"${PROMPT:0:50}...\""
echo "   Tokens to generate: $N_TOKENS"
echo ""

# Function to extract tokens per second from output
extract_tps() {
    grep -E "tok/s|tokens per second|ms/tok" "$1" | tail -1
}

# Test 1: CPU baseline
echo "1️⃣ Testing CPU performance..."
echo "   Command: $LLAMA_CLI -m \"$MODEL\" -p \"$PROMPT\" -n $N_TOKENS --no-gpu"
$LLAMA_CLI -m "$MODEL" -p "$PROMPT" -n $N_TOKENS --no-gpu 2>&1 | tee cpu_benchmark.log
CPU_RESULT=$(extract_tps cpu_benchmark.log)
echo "   Result: $CPU_RESULT"
echo ""

# Test 2: Vulkan GPU (if available)
if $LLAMA_CLI --help 2>&1 | grep -q "gpu-layers"; then
    echo "2️⃣ Testing Vulkan GPU performance..."
    echo "   Command: $LLAMA_CLI -m \"$MODEL\" -p \"$PROMPT\" -n $N_TOKENS --gpu-layers 999"
    $LLAMA_CLI -m "$MODEL" -p "$PROMPT" -n $N_TOKENS --gpu-layers 999 2>&1 | tee gpu_benchmark.log
    GPU_RESULT=$(extract_tps gpu_benchmark.log)
    echo "   Result: $GPU_RESULT"
else
    echo "2️⃣ Vulkan GPU support not available in this build"
    GPU_RESULT="N/A"
fi
echo ""

# Test 3: NPU acceleration
if $LLAMA_CLI --help 2>&1 | grep -q "npu-attention"; then
    echo "3️⃣ Testing NPU acceleration..."
    echo "   Command: $LLAMA_CLI -m \"$MODEL\" -p \"$PROMPT\" -n $N_TOKENS --npu-attention"
    $LLAMA_CLI -m "$MODEL" -p "$PROMPT" -n $N_TOKENS --npu-attention 2>&1 | tee npu_benchmark.log
    NPU_RESULT=$(extract_tps npu_benchmark.log)
    echo "   Result: $NPU_RESULT"
else
    echo "3️⃣ NPU support not available in this build"
    NPU_RESULT="N/A"
fi
echo ""

# Test 4: Combined NPU + GPU (if both available)
if $LLAMA_CLI --help 2>&1 | grep -q "npu-attention" && $LLAMA_CLI --help 2>&1 | grep -q "gpu-layers"; then
    echo "4️⃣ Testing NPU + GPU combined..."
    echo "   Command: $LLAMA_CLI -m \"$MODEL\" -p \"$PROMPT\" -n $N_TOKENS --npu-attention --gpu-layers 999"
    $LLAMA_CLI -m "$MODEL" -p "$PROMPT" -n $N_TOKENS --npu-attention --gpu-layers 999 2>&1 | tee combined_benchmark.log
    COMBINED_RESULT=$(extract_tps combined_benchmark.log)
    echo "   Result: $COMBINED_RESULT"
else
    COMBINED_RESULT="N/A"
fi
echo ""

# Summary
echo "📊 PERFORMANCE SUMMARY"
echo "===================="
echo "CPU Baseline:    $CPU_RESULT"
echo "Vulkan GPU:      $GPU_RESULT"
echo "NPU Attention:   $NPU_RESULT"
echo "NPU + GPU:       $COMBINED_RESULT"
echo ""

# Save results
cat > benchmark_results.json << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "model": "$MODEL",
  "tokens_generated": $N_TOKENS,
  "results": {
    "cpu": "$CPU_RESULT",
    "vulkan_gpu": "$GPU_RESULT",
    "npu": "$NPU_RESULT",
    "npu_gpu_combined": "$COMBINED_RESULT"
  }
}
EOF

echo "💾 Results saved to benchmark_results.json"
echo ""
echo "✅ Benchmark complete!"