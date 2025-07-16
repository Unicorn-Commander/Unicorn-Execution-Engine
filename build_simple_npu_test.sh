#!/bin/bash
# Build Simple NPU Test - Focus on C++ Engine Performance

set -e

echo "🔥 Building Simple NPU Performance Test"
echo "======================================="

# Use ROCm clang++
export CXX="/opt/rocm/llvm/bin/clang++"

echo "🔧 Compiler: $CXX"

echo ""
echo "📊 Building optimized C++ NPU execution engine..."

# Build with maximum optimization
$CXX -O3 -fPIC -shared \
    -fopenmp \
    -march=native \
    -mavx2 \
    -mfma \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    real_npu_execution.cpp \
    -o libreal_npu_engine.so

echo "✅ Optimized C++ NPU execution engine built: libreal_npu_engine.so"

echo ""
echo "🧪 Testing C++ engine loading..."

# Test if the library can be loaded
python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('./libreal_npu_engine.so')
    print('✅ Library loads successfully')
    print(f'   Library: {lib}')
except Exception as e:
    print(f'❌ Library load failed: {e}')
"

echo ""
echo "🎉 SIMPLE NPU TEST BUILD COMPLETE!"
echo "=================================="
echo ""
echo "📁 Generated files:"
echo "   🔥 libreal_npu_engine.so - Optimized C++ execution engine"
echo ""
echo "🚀 Ready for performance testing!"
echo "   • AVX2 + FMA optimization enabled"
echo "   • OpenMP parallel processing"
echo "   • Native architecture targeting"
echo ""
echo "🔧 Test with:"
echo "   python real_npu_performance_test.py"
echo ""