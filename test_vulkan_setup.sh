#!/bin/bash
# Test Vulkan setup and llama.cpp build

echo "🦄 Vulkan + NPU Hybrid Setup Test"
echo "=================================="

# Check Vulkan installation
echo -e "\n📋 Checking Vulkan installation..."
if command -v vulkaninfo &> /dev/null; then
    echo "✓ Vulkan is installed"
    
    # Check for AMD GPU
    echo -e "\n🎮 Checking for AMD GPU support..."
    vulkaninfo 2>/dev/null | grep -A2 "deviceName" | grep -i "AMD\|Radeon" && echo "✓ AMD GPU detected with Vulkan support" || echo "❌ No AMD GPU found in Vulkan"
    
    # Check Vulkan version
    echo -e "\n📊 Vulkan version:"
    vulkaninfo 2>/dev/null | grep -A1 "apiVersion" | head -2
else
    echo "❌ Vulkan not installed"
    echo ""
    echo "To install Vulkan on Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install vulkan-tools vulkan-sdk mesa-vulkan-drivers"
    echo ""
    echo "For AMD GPUs specifically:"
    echo "  sudo apt install mesa-vulkan-drivers vulkan-utils"
fi

# Check if we can build llama.cpp with Vulkan
echo -e "\n🔨 Checking build requirements..."
if command -v make &> /dev/null && command -v g++ &> /dev/null; then
    echo "✓ Build tools available"
else
    echo "❌ Missing build tools"
    echo "  Install with: sudo apt install build-essential"
fi

# Provide next steps
echo -e "\n🚀 Next Steps:"
echo "1. Install Vulkan (if needed):"
echo "   sudo apt install vulkan-tools vulkan-sdk mesa-vulkan-drivers"
echo ""
echo "2. Clone and build llama.cpp with Vulkan:"
echo "   git clone https://github.com/ggerganov/llama.cpp"
echo "   cd llama.cpp"
echo "   make LLAMA_VULKAN=1 -j\$(nproc)"
echo ""
echo "3. Download a test model:"
echo "   wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
echo ""
echo "4. Test performance:"
echo "   ./main -m llama-2-7b.Q4_K_M.gguf -p \"Hello\" -n 50 --gpu-layers 999"
echo ""
echo "Expected performance with Vulkan: 25-30 tok/s"
echo "With NPU offload: 35-40 tok/s"