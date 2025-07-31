#!/bin/bash
# Download Gemma 2 9B Q4 model (similar size class to Gemma 3n but Q4 quantized)

MODEL_URL="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
MODEL_NAME="gemma-2-9b-it-Q4_K_M.gguf"

echo "📥 Downloading Gemma 2 9B Q4_K_M model..."
echo "This is a 9B parameter model with Q4 quantization"
echo "Expected size: ~5.5GB"
echo ""

# Download with resume capability
wget -c "$MODEL_URL" -O "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Download complete!"
    ls -lh "$MODEL_NAME"
else
    echo "❌ Download failed"
fi