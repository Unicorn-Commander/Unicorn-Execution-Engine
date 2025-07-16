#!/bin/bash

# AMD Native Gemma 3 27B API Server Startup Script
# Port: 8005, Host: 0.0.0.0
# Features: Direct Vulkan + XRT + AMDgpu (no PyTorch CUDA)

set -e

echo "🎮 AMD NATIVE GEMMA 3 27B API SERVER"
echo "============================================================"
echo "🚀 HARDWARE: Direct Vulkan compute + XRT/XDNA + AMDgpu memory"
echo "⚡ ACCELERATION: 100% AMD native (bypasses PyTorch CUDA entirely)"
echo "💾 MEMORY: Direct VRAM/GTT allocation via AMDgpu APIs"
echo "🔤 PROCESSING: Vulkan shaders + XRT kernels"
echo "📡 Server: http://0.0.0.0:8005"
echo "⏱️ Startup: 5-10 seconds for AMD hardware initialization"
echo "🎯 Result: Pure AMD hardware acceleration!"
echo

# Check if we're in the correct directory
if [ ! -f "amd_native_api_server.py" ]; then
    echo "❌ Error: amd_native_api_server.py not found in current directory"
    echo "Please run this script from the Unicorn-Execution-Engine directory"
    exit 1
fi

# Activate the Python environment
echo "🐍 Activating Python environment..."
if [ -f "/home/ucadmin/activate-uc1-ai-py311.sh" ]; then
    source /home/ucadmin/activate-uc1-ai-py311.sh
    echo "✅ Environment activated"
else
    echo "⚠️ Warning: Environment activation script not found"
    echo "Continuing with current Python environment..."
fi

# Check AMD hardware
echo "🔧 Checking AMD hardware availability..."

# Check Vulkan
if command -v vulkaninfo &> /dev/null; then
    echo "✅ Vulkan available"
else
    echo "⚠️ Warning: vulkaninfo not found"
fi

# Check XRT/XDNA
if command -v xrt-smi &> /dev/null; then
    echo "✅ XRT available"
    xrt-smi examine | grep -i phoenix || echo "⚠️ NPU Phoenix not detected"
else
    echo "⚠️ Warning: xrt-smi not found"
fi

# Check ROCm/AMDgpu
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm available"
    rocm-smi --showproductname | grep -i 780M || echo "⚠️ AMD Radeon 780M not detected"
else
    echo "⚠️ Warning: rocm-smi not found"
fi

# Check if model directory exists
MODEL_DIR="./quantized_models/gemma-3-27b-it-layer-by-layer"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    echo "Please ensure the quantized Gemma 3 27B model is available"
    exit 1
fi

echo "📁 Model directory: $MODEL_DIR ✅"

# Start the server
echo "🚀 Starting AMD Native Gemma 3 27B API Server..."
echo "🎮 VULKAN: Direct iGPU compute shaders"
echo "⚡ XRT/XDNA: Direct NPU kernel execution"
echo "💾 AMDGPU: Direct VRAM/GTT memory management"
echo "⏱️ STARTUP TIME: Expect 5-10 seconds for AMD hardware init"
echo "🔗 URL: http://0.0.0.0:8005"
echo "🛑 Press Ctrl+C to stop the server"
echo

python amd_native_api_server.py