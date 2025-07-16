#!/bin/bash

echo "🧪 Testing Fixed NPU+iGPU Server"
echo "================================"

# 1. Verify MLIR-AIE2 environment
echo "📋 1. Testing MLIR-AIE2 Environment..."
source /home/ucadmin/mlir-aie2/ironenv/bin/activate
python -c "
from aie.iron import ObjectFifo
from aie.iron.device import NPU1Col1
print('✅ MLIR-AIE2 imports working')
print(f'✅ ObjectFifo: {ObjectFifo}')
print(f'✅ NPU1Col1: {NPU1Col1}')
"
deactivate

echo ""
echo "📋 2. Testing Hardware Detection..."
echo "NPU Phoenix:"
xrt-smi examine | grep -i phoenix || echo "❌ NPU not detected"

echo "AMD Radeon 780M:"
vulkaninfo --summary | grep -i "amd radeon" || echo "❌ iGPU not detected"

echo ""
echo "📋 3. Starting Server..."
echo "Expected behavior:"
echo "  ✅ Should find MLIR-AIE2 at correct path"
echo "  ✅ Should require NPU+iGPU or fail (no fallbacks)"
echo "  ✅ Should show INSTANT ACCESS during layer loading"
echo "  ✅ Layer loading should be <100ms each"

echo ""
echo "🚀 Starting real_2025_gemma27b_server.py..."
echo "Press Ctrl+C to stop server when ready to test"
echo ""

# Activate correct environment
source /home/ucadmin/activate-uc1-ai-py311.sh

# Run server
python real_2025_gemma27b_server.py