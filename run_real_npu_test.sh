#!/bin/bash
"""
Run Real NPU Performance Test
Complete script to test real NPU performance for Gemma 3 27B
"""

echo "🚀 Real NPU Performance Test for Gemma 3 27B"
echo "=============================================="
echo ""

# Activate AI environment
echo "🔧 Activating AI environment..."
source ~/activate-uc1-ai-py311.sh

# Check if we're in the right directory
if [ ! -f "real_npu_performance_test.py" ]; then
    echo "❌ Error: Must run from Unicorn-Execution-Engine directory"
    echo "   cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine"
    exit 1
fi

echo "✅ Environment activated"
echo ""

# Step 1: Verify hardware setup
echo "🔍 Step 1: Verifying hardware setup..."
python verify_real_hardware_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Hardware verification failed!"
    echo "   Fix the issues above before continuing"
    exit 1
fi

echo ""
echo "✅ Hardware verification passed!"
echo ""

# Step 2: Compile NPU kernels (if needed)
echo "🔧 Step 2: Ensuring NPU kernels are compiled..."
if [ ! -d "npu_binaries" ] || [ ! -f "npu_binaries/gemma3_q_projection.npu_binary" ]; then
    echo "🔥 Compiling NPU kernels..."
    source mlir-aie2-src/ironenv/bin/activate
    python compile_npu_kernels.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Kernel compilation failed!"
        exit 1
    fi
    
    echo "✅ NPU kernels compiled"
else
    echo "✅ NPU kernels already compiled"
fi

echo ""

# Step 3: Setup test data
echo "📊 Step 3: Setting up test data..."
python setup_real_model_test.py

if [ $? -ne 0 ]; then
    echo "❌ Test data setup failed!"
    exit 1
fi

echo "✅ Test data ready"
echo ""

# Step 4: Run real NPU performance test
echo "🚀 Step 4: Running REAL NPU performance test..."
echo "⚡ This will test actual NPU hardware performance"
echo "💡 Expected to take 2-5 minutes for complete testing"
echo ""

python real_npu_performance_test.py

test_result=$?

echo ""
echo "=============================================="

if [ $test_result -eq 0 ]; then
    echo "🎉 REAL NPU PERFORMANCE TEST COMPLETED!"
    echo ""
    echo "📊 Results saved to: real_npu_performance_results.json"
    echo ""
    echo "🚀 Check the log output above for tokens/second performance"
    echo "✅ All tests completed successfully"
else
    echo "❌ Real NPU performance test failed!"
    echo "   Check the error messages above"
fi

echo "=============================================="