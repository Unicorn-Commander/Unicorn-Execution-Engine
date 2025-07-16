#!/bin/bash
# Start Pure Hardware API Server with Proper GPU Memory Allocation

echo "🦄 STARTING PURE HARDWARE INFERENCE SERVER"
echo "========================================"
echo ""
echo "This server uses:"
echo "  ✅ Vulkan for VRAM/GTT allocation (not system RAM)"
echo "  ✅ NPU kernels for attention computation"
echo "  ✅ Direct hardware acceleration"
echo "  ✅ No PyTorch/ROCm dependencies"
echo ""

# Activate environment
echo "🔧 Activating environment..."
source /home/ucadmin/activate-uc1-ai-py311.sh

# Check GPU memory before starting
echo "📊 Initial GPU Memory:"
rocm-smi --showmeminfo vram | grep -E "Total|Used"
echo ""

# Check if cffi is installed (needed for Vulkan)
python -c "import cffi" 2>/dev/null || {
    echo "Installing cffi for Vulkan support..."
    pip install cffi
}

# Start the server
echo "🚀 Starting server on port 8008..."
echo ""
python pure_hardware_api_server_final.py