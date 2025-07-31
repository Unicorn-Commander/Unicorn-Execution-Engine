#!/bin/bash
# Download a small test model for llama.cpp

echo "🦄 Downloading test model..."

# Try different sources
MODEL_URL="https://huggingface.co/LiteLLMs/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf"
MODEL_NAME="llama-3.2-1b-q4_k_m.gguf"

# Alternative: Use a smaller test model
if ! wget -O "$MODEL_NAME" "$MODEL_URL" 2>/dev/null; then
    echo "Primary download failed, trying alternative..."
    
    # Try Phi-2 which is smaller
    MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    MODEL_NAME="phi-2-q4_k_m.gguf"
    
    if ! wget -O "$MODEL_NAME" "$MODEL_URL" 2>/dev/null; then
        echo "❌ Could not download model. Please download manually."
        echo "Suggested models:"
        echo "  - https://huggingface.co/models?search=gguf+q4_k_m"
        exit 1
    fi
fi

echo "✅ Model downloaded: $MODEL_NAME"
echo "Size: $(ls -lh $MODEL_NAME | awk '{print $5}')"