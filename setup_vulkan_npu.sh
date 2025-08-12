#!/bin/bash
# Automated setup script for Vulkan + NPU integration with llama.cpp

set -e  # Exit on error

echo "🦄 Vulkan + NPU Integration Setup"
echo "================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check for required tools
    local missing_tools=()
    
    for tool in cmake gcc g++ git wget make; do
        if ! command -v $tool &> /dev/null; then
            missing_tools+=($tool)
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing tools: ${missing_tools[*]}${NC}"
        echo "Install with: sudo apt install ${missing_tools[*]}"
        return 1
    fi
    
    # Check Vulkan
    if ! command -v vulkaninfo &> /dev/null; then
        echo -e "${YELLOW}⚠️  Vulkan tools not found${NC}"
        echo "Installing Vulkan dependencies..."
        sudo apt update
        sudo apt install -y libvulkan-dev vulkan-tools glslc
    fi
    
    # Check NPU
    if [ ! -e "/dev/accel/accel0" ]; then
        echo -e "${YELLOW}⚠️  NPU device not found${NC}"
        echo "Attempting to load NPU driver..."
        sudo modprobe amdxdna aie2_control_flags=7 || true
    fi
    
    # Check XRT
    if [ ! -d "/opt/xilinx/xrt" ]; then
        echo -e "${RED}❌ XRT not installed${NC}"
        echo "Please install XRT for NPU support"
        return 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites satisfied${NC}"
    return 0
}

# Build llama.cpp with Vulkan
build_llama_vulkan() {
    echo -e "\n🔨 Building llama.cpp with Vulkan..."
    
    if [ ! -d "llama.cpp" ]; then
        echo "Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp
    fi
    
    cd llama.cpp
    
    # Clean previous builds
    rm -rf build
    
    # Build with Vulkan
    echo "Configuring with Vulkan support..."
    cmake -B build \
        -DGGML_VULKAN=ON \
        -DCMAKE_BUILD_TYPE=Release
    
    echo "Building..."
    cmake --build build --config Release -j$(nproc)
    
    # Verify Vulkan support
    if ./build/bin/llama-cli --help 2>&1 | grep -q vulkan; then
        echo -e "${GREEN}✅ Vulkan support confirmed${NC}"
    else
        echo -e "${RED}❌ Vulkan support not detected${NC}"
        return 1
    fi
    
    cd ..
    return 0
}

# Build NPU backend
build_npu_backend() {
    echo -e "\n🔨 Building NPU backend..."
    
    cd llama-npu-integration
    
    # Clean and build
    rm -rf build
    mkdir build && cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NPU_BUILD_TESTS=ON
    
    make -j$(nproc)
    
    # Run tests
    echo -e "\n🧪 Running NPU tests..."
    if ./test-npu; then
        echo -e "${GREEN}✅ NPU tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  NPU tests failed (NPU may not be available)${NC}"
    fi
    
    cd ../..
    return 0
}

# Download test model
download_test_model() {
    echo -e "\n📥 Downloading test model..."
    
    # Try to download a small model
    if [ ! -f "tinyllama-1.1b-q4_k_m.gguf" ]; then
        echo "Downloading TinyLlama 1.1B Q4_K_M..."
        wget -q --show-progress \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
            -O tinyllama-1.1b-q4_k_m.gguf || {
                echo -e "${YELLOW}⚠️  Could not download model${NC}"
                echo "Please download a GGUF model manually"
                return 1
            }
    fi
    
    echo -e "${GREEN}✅ Model ready: tinyllama-1.1b-q4_k_m.gguf${NC}"
    return 0
}

# Create benchmark scripts
create_scripts() {
    echo -e "\n📝 Creating helper scripts..."
    
    # Vulkan-only benchmark
    cat > benchmark_vulkan.sh << 'EOF'
#!/bin/bash
MODEL=${1:-tinyllama-1.1b-q4_k_m.gguf}

echo "🚀 Benchmarking Vulkan-only performance"
echo "Model: $MODEL"
echo ""

# CPU baseline
echo "1️⃣ CPU Baseline:"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "The future of AI is" -n 50 \
    --no-display-prompt --gpu-layers 0 2>&1 | \
    grep -E "(tok/s|tokens per second)"

# Vulkan GPU
echo -e "\n2️⃣ Vulkan GPU:"
./llama.cpp/build/bin/llama-cli -m "$MODEL" \
    -p "The future of AI is" -n 50 \
    --no-display-prompt --gpu-layers 999 2>&1 | \
    grep -E "(tok/s|tokens per second)"
EOF

    # NPU integration test (future)
    cat > test_npu_integration.sh << 'EOF'
#!/bin/bash
MODEL=${1:-tinyllama-1.1b-q4_k_m.gguf}

echo "🦄 Testing NPU Integration"
echo "Model: $MODEL"
echo ""

# Check NPU status
echo "NPU Status:"
/opt/xilinx/xrt/bin/xrt-smi examine 2>/dev/null | grep -E "(Device|AIE)" || echo "NPU not available"

# Run NPU backend tests
echo -e "\nNPU Backend Tests:"
./llama-npu-integration/build/test-npu

# Benchmark NPU operations
echo -e "\nNPU Benchmark:"
./llama-npu-integration/build/benchmark-npu
EOF

    chmod +x benchmark_vulkan.sh test_npu_integration.sh
    
    echo -e "${GREEN}✅ Scripts created${NC}"
}

# Print summary
print_summary() {
    echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
    echo "=================="
    echo ""
    echo "📊 Performance Expectations:"
    echo "  • CPU only: 1-5 tok/s"
    echo "  • Vulkan: 25-30 tok/s"
    echo "  • Vulkan+NPU: 35-40 tok/s (after full integration)"
    echo ""
    echo "🚀 Quick Start:"
    echo "  1. Test Vulkan:     ./benchmark_vulkan.sh"
    echo "  2. Test NPU:        ./test_npu_integration.sh"
    echo "  3. Run inference:   ./llama.cpp/build/bin/llama-cli -m tinyllama-1.1b-q4_k_m.gguf -p \"Hello\" --gpu-layers 999"
    echo ""
    echo "📚 Documentation:"
    echo "  • Integration guide: llama-npu-integration/NPU_LLAMA_INTEGRATION.md"
    echo "  • Architecture:      llama-npu-integration/INTEGRATION_GUIDE.md"
    echo ""
    echo "⚠️  Note: Full NPU integration requires manual modification of llama.cpp"
    echo "         See integration guide for details."
}

# Main execution
main() {
    echo "Starting setup in: $(pwd)"
    echo ""
    
    # Run setup steps
    check_prerequisites || exit 1
    build_llama_vulkan || exit 1
    build_npu_backend || exit 1
    download_test_model || echo "Continuing without model..."
    create_scripts
    
    # Final summary
    print_summary
}

# Run main
main