#!/bin/bash
# Download official Google Gemma 3 27B Q4_0 model

MODEL_URL="https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf/resolve/main/gemma-3-27b-it-qat-q4_0.gguf"
MODEL_NAME="gemma-3-27b-it-qat-q4_0.gguf"

echo "📥 Downloading Google Gemma 3 27B Q4_0 model..."
echo "This is the official 27B parameter model with Q4_0 quantization"
echo "Expected size: ~15-16GB"
echo ""
echo "Model: google/gemma-3-27b-it-qat-q4_0-gguf"
echo "URL: $MODEL_URL"
echo ""

# Check disk space
echo "💾 Current disk space:"
df -h . | tail -1
echo ""

# Download with resume capability
echo "🚀 Starting download (this may take a while)..."
wget -c "$MODEL_URL" -O "$MODEL_NAME" --progress=dot:giga

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    ls -lh "$MODEL_NAME"
    
    echo ""
    echo "🔍 Model info:"
    file "$MODEL_NAME"
else
    echo "❌ Download failed"
    echo "You can resume the download by running this script again"
fi