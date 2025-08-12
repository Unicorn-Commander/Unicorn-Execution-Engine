#!/bin/bash
# Build a simple test to get real performance numbers

echo "🚀 Building simple llama test..."

cd llama.cpp

# Since CMake is having issues, let's try the Makefile approach
if [ -f "Makefile" ]; then
    echo "📦 Found Makefile, attempting direct build..."
    
    # Set environment for XRT
    export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
    export CXXFLAGS="-I/opt/xilinx/xrt/include -DLLAMA_NPU_XRT_ENABLED"
    export LDFLAGS="-L/opt/xilinx/xrt/lib -lxrt++ -lxrt_core -lxrt_coreutil"
    
    # Try to build with make
    make clean 2>/dev/null
    make GGML_VULKAN=1 -j4 llama-cli 2>&1 | tail -50
    
    if [ -f "llama-cli" ]; then
        echo "✅ Build successful!"
        echo "📍 Binary at: ./llama-cli"
        
        # Test it
        echo ""
        echo "🧪 Testing NPU support..."
        ./llama-cli --help 2>&1 | grep -i npu || echo "⚠️ NPU flag not visible"
        
        # Create a test runner
        cat > ../run_npu_benchmark.sh << 'EOF'
#!/bin/bash
cd llama.cpp
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH

echo "🏃 Running NPU benchmark..."
echo "Model: gemma-3n-E4B-it-Q8_0.gguf"
echo ""

# Run with NPU
./llama-cli -m ../gemma-3n-E4B-it-Q8_0.gguf \
    -p "Once upon a time in a magical forest, there lived a wise old owl" \
    -n 50 \
    --npu-attention \
    2>&1 | tee npu_benchmark_output.log

# Extract performance
echo ""
echo "📊 Performance Results:"
grep -E "(tok/s|tokens per second|ms/tok)" npu_benchmark_output.log
EOF
        chmod +x ../run_npu_benchmark.sh
        echo "✅ Created benchmark script: ../run_npu_benchmark.sh"
    else
        echo "❌ Make build failed"
    fi
else
    echo "❌ No Makefile found"
fi

# If all else fails, let's check what we have
echo ""
echo "🔍 Available executables:"
find . -name "llama*" -type f -executable | head -10