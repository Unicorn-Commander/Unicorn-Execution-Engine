#!/bin/bash
# Download additional Gemma models for testing

echo "🦄 Gemma Model Downloader"
echo "========================"

# Check available disk space
echo -e "\n📊 Disk space:"
df -h . | tail -1

echo -e "\n📦 Available models to download:"
echo "1. Gemma 3n Q4_K_M (smaller, faster)"
echo "2. Gemma 7B Q4_K_M (medium size)"
echo "3. Gemma 27B Q4_K_M (large, if available)"

# Common HuggingFace repos for GGUF models
echo -e "\n🔍 Searching for model URLs..."

# Function to download model
download_model() {
    local url=$1
    local filename=$2
    
    if [ -f "$filename" ]; then
        echo "✅ $filename already exists"
        return
    fi
    
    echo "📥 Downloading $filename..."
    wget -c "$url" -O "$filename" || {
        echo "❌ Download failed"
        rm -f "$filename"
        return 1
    }
    
    echo "✅ Downloaded $filename"
}

# Check HuggingFace for Gemma models
echo -e "\n🌐 Checking available Gemma GGUF models..."

# Common repos that host GGUF versions
repos=(
    "TheBloke"
    "NousResearch" 
    "mlabonne"
    "bartowski"
)

echo -e "\n💡 Suggested downloads:"
echo ""
echo "# For Gemma 3n Q4:"
echo "wget https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
echo ""
echo "# For larger Gemma models:"
echo "wget https://huggingface.co/mradermacher/Gemma-2-Ataraxy-v4d-27B-GGUF/resolve/main/Gemma-2-Ataraxy-v4d-27B.Q4_K_M.gguf"
echo ""
echo "# Alternative Q4 models:"
echo "wget https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

echo -e "\n📝 Note: These are large files (2-15GB each)"
echo "Use wget -c to resume interrupted downloads"