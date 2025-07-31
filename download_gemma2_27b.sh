#!/bin/bash
# Download Gemma 2 27B Q4_K_M model (community version, no auth required)

MODEL_URL="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/resolve/main/gemma-2-27b-it-Q4_K_M.gguf"
MODEL_NAME="gemma-2-27b-it-Q4_K_M.gguf"

echo "📥 Downloading Gemma 2 27B Q4_K_M model (Community version)..."
echo "This is a 27B parameter model with Q4_K_M quantization"
echo "Expected size: ~16GB"
echo ""
echo "Model: bartowski/gemma-2-27b-it-GGUF"
echo "URL: $MODEL_URL"
echo ""

# Check disk space
echo "💾 Current disk space:"
df -h . | tail -1
echo ""

# Download with resume capability
echo "🚀 Starting download (this will take a while)..."
echo "Press Ctrl+C to pause - you can resume later by running this script again"
echo ""
wget -c "$MODEL_URL" -O "$MODEL_NAME" --show-progress

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    ls -lh "$MODEL_NAME"
    
    echo ""
    echo "📊 To test this model:"
    echo "./llama.cpp/build/bin/llama-cli -m $MODEL_NAME -p \"Hello, tell me about AI\" -n 50 --n-gpu-layers 999"
else
    echo ""
    echo "⚠️  Download interrupted or failed"
    echo "Run this script again to resume the download"
fi