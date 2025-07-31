#!/bin/bash
# Activation script for Magic Unicorn Python 3.13 environment

echo "🦄 Activating Magic Unicorn Hardware-Only Environment"
source /home/ucadmin/Development/Unicorn-Execution-Engine/magic-unicorn-env/bin/activate

# Set environment variables for NPU and GPU
export XILINX_XRT=/opt/xilinx/xrt
export PYTHONPATH=/opt/xilinx/xrt/python:/home/ucadmin/Development/Unicorn-Execution-Engine:$PYTHONPATH
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
export TORCH_USE_CUDA_DSA=0  # We don't use torch anyway

echo "✅ Environment ready!"
echo "   Python: $(python --version)"
echo "   NPU: XRT at $XILINX_XRT"
echo "   GPU: Vulkan with RADV driver"
echo ""
echo "Run: python launch_hardware_only.py"