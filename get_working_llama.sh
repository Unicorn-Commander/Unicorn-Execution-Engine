#!/bin/bash
# Get a working llama.cpp build

echo "🚀 Getting a working llama.cpp build..."
echo "======================================"

# Option 1: Check if ollama is installed (it uses llama.cpp internally)
if command -v ollama &> /dev/null; then
    echo "✅ Found ollama installation"
    echo "   You can use: ollama run gemma:2b"
    echo ""
fi

# Option 2: Download pre-built llama.cpp
echo "📦 Downloading pre-built llama.cpp..."
cd /home/ucadmin/Development/Unicorn-Execution-Engine

# Get the latest release
RELEASE_URL="https://github.com/ggerganov/llama.cpp/releases/latest"
echo "🔍 Checking latest release..."

# Download a pre-built binary if available
wget -q -O llama-cli "https://github.com/ggerganov/llama.cpp/releases/download/b3166/llama-b3166-bin-ubuntu-x64.zip" 2>/dev/null || true

# Option 3: Clone a fresh copy and build
if [ ! -f "llama-cli" ]; then
    echo "📥 Cloning fresh llama.cpp..."
    rm -rf llama.cpp.fresh
    git clone https://github.com/ggerganov/llama.cpp llama.cpp.fresh
    cd llama.cpp.fresh
    
    # Simple build
    mkdir build
    cd build
    cmake .. -DGGML_VULKAN=ON
    make -j4 llama-cli
    
    if [ -f "bin/llama-cli" ]; then
        echo "✅ Fresh build successful!"
        cp bin/llama-cli ../../llama-cli-fresh
        cd ../..
        echo "📍 Binary: ./llama-cli-fresh"
    fi
fi

# Option 4: Use the existing NPU-integrated version with LD_PRELOAD
echo ""
echo "💡 Alternative: Use existing build with XRT preloaded"
cat > run_with_xrt.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="/opt/xilinx/xrt/lib/libxrt_core.so:/opt/xilinx/xrt/lib/libxrt++.so"

# Find any llama binary
LLAMA=$(find . -name "llama-cli" -type f -executable | head -1)
if [ -z "$LLAMA" ]; then
    LLAMA=$(find . -name "llama-simple" -type f -executable | head -1)
fi

if [ -n "$LLAMA" ]; then
    echo "🚀 Running $LLAMA with XRT libraries preloaded"
    exec "$LLAMA" "$@"
else
    echo "❌ No llama binary found"
fi
EOF
chmod +x run_with_xrt.sh

echo ""
echo "✅ Options available:"
echo "   1. ./run_with_xrt.sh -m model.gguf -p \"prompt\" --npu-attention"
echo "   2. ./llama-cli-fresh -m model.gguf -p \"prompt\" (if fresh build worked)"
echo "   3. Use ollama if installed"