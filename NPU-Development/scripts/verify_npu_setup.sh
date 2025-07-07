#!/bin/bash

# NPU Setup Verification Script
# Comprehensive verification of NPU development environment



# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    log_info "Testing: $test_name"
    
    if eval "$test_command" &>/dev/null; then
        log_success "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "$test_name"
        FAILED_TESTS+=("$test_name")
        ((TESTS_FAILED++))
        return 1
    fi
}

# Detailed test with output
run_detailed_test() {
    local test_name="$1"
    local test_command="$2"
    
    log_info "Testing: $test_name"
    
    local output
    if output=$(eval "$test_command" 2>&1); then
        log_success "$test_name"
        echo "  Output: $output"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "$test_name"
        echo "  Error: $output"
        FAILED_TESTS+=("$test_name")
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "=== NPU Development Environment Verification ==="
echo "Starting comprehensive NPU setup verification..."
echo

# 1. Hardware Detection Tests
echo "1. Hardware Detection"
echo "===================="

run_test "NPU Device in lspci" "lspci | grep -qi 'signal processing'"
run_test "NPU Device Files" "test -e /dev/accel/accel0"
run_test "Multiple NPU Devices" "ls /dev/accel/ | grep -q accel"

echo

# 2. Kernel Driver Tests
echo "2. Kernel Driver"
echo "==============="

run_test "XDNA Driver Loaded" "lsmod | grep -q amdxdna"
run_test "Driver Version Check" "modinfo amdxdna | grep -q version"
run_detailed_test "Driver Device Detection" "cat /proc/modules | grep amdxdna"

echo

# 3. XRT Environment Tests
echo "3. XRT (Xilinx Runtime)"
echo "======================"

run_test "XRT Installation" "test -f /opt/xilinx/xrt/setup.sh"
run_test "XRT Environment" "source /opt/xilinx/xrt/setup.sh && command -v xrt-smi"
run_detailed_test "XRT NPU Detection" "source /opt/xilinx/xrt/setup.sh && xrt-smi examine | grep -i npu"

echo

# 4. MLIR-AIE Framework Tests
echo "4. MLIR-AIE Framework"
echo "===================="

# Check for MLIR-AIE installation
if [ -d "$HOME/npu-dev/mlir-aie" ]; then
    MLIR_AIE_PATH="$HOME/npu-dev/mlir-aie"
elif [ -d "./mlir-aie" ]; then
    MLIR_AIE_PATH="./mlir-aie"
else
    MLIR_AIE_PATH=""
fi

if [ -n "$MLIR_AIE_PATH" ]; then
    run_test "MLIR-AIE Installation" "test -d $MLIR_AIE_PATH"
    run_test "MLIR-AIE Python Environment" "test -d $MLIR_AIE_PATH/ironenv"
    run_test "MLIR-AIE Build Directory" "test -d $MLIR_AIE_PATH/build"
    
    # Test Python imports
    if [ -d "$MLIR_AIE_PATH/ironenv" ]; then
        run_detailed_test "MLIR-AIE Python Import" \
            "source $MLIR_AIE_PATH/ironenv/bin/activate && python -c 'import mlir_aie; print(\"MLIR-AIE import successful\")'"
    fi
else
    log_warning "MLIR-AIE not found in expected locations"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("MLIR-AIE Installation")
fi

echo

# 5. Python Environment Tests
echo "5. Python Environment"
echo "===================="

# Check for NPU development environment
if [ -d "$HOME/npu-dev/npu_dev_env" ]; then
    NPU_ENV_PATH="$HOME/npu-dev/npu_dev_env"
elif [ -d "./npu_dev_env" ]; then
    NPU_ENV_PATH="./npu_dev_env"
else
    NPU_ENV_PATH=""
fi

if [ -n "$NPU_ENV_PATH" ]; then
    run_test "NPU Python Environment" "test -d $NPU_ENV_PATH"
    
    # Test Python packages
    run_detailed_test "ONNX Runtime" \
        "source $NPU_ENV_PATH/bin/activate && python -c 'import onnxruntime; print(f\"ONNX Runtime {onnxruntime.__version__}\")'"
    
    run_detailed_test "PyTorch" \
        "source $NPU_ENV_PATH/bin/activate && python -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
    
    run_detailed_test "NumPy" \
        "source $NPU_ENV_PATH/bin/activate && python -c 'import numpy; print(f\"NumPy {numpy.__version__}\")'"
    
    run_detailed_test "Transformers" \
        "source $NPU_ENV_PATH/bin/activate && python -c 'import transformers; print(f\"Transformers {transformers.__version__}\")'"
else
    log_warning "NPU Python environment not found"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("NPU Python Environment")
fi

echo

# 6. NPU Functional Tests
echo "6. NPU Functional Tests"
echo "======================"

# Test basic NPU functionality
run_detailed_test "NPU Device Memory" "cat /sys/class/accel/accel0/device/mem_info 2>/dev/null || echo 'Memory info not available'"

# Test NPU with simple operation (if custom kernels are available)
if [ -f "npu_kernels/matrix_multiply.py" ]; then
    run_test "NPU Kernel Test" "python npu_kernels/matrix_multiply.py --test"
else
    log_warning "NPU kernel tests not available (npu_kernels/matrix_multiply.py not found)"
fi

echo

# 7. Performance Verification
echo "7. Performance Verification"
echo "==========================="

# Basic performance tests
if [ -n "$NPU_ENV_PATH" ]; then
    # Create simple performance test
    cat > /tmp/npu_perf_test.py << 'EOF'
import time
import numpy as np

def test_matrix_multiply():
    """Test matrix multiplication performance"""
    size = 512
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    start_time = time.perf_counter()
    c = np.dot(a, b)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    print(f"Matrix multiplication ({size}x{size}): {duration:.3f}s")
    return duration < 1.0  # Should complete within 1 second

if __name__ == "__main__":
    success = test_matrix_multiply()
    exit(0 if success else 1)
EOF

    run_detailed_test "CPU Baseline Performance" \
        "source $NPU_ENV_PATH/bin/activate && python /tmp/npu_perf_test.py"
    
    # Clean up
    rm -f /tmp/npu_perf_test.py
fi

echo

# 8. Integration Tests
echo "8. Integration Tests"
echo "==================="

# Test environment setup script
if [ -f "$HOME/npu-dev/setup_npu_env.sh" ]; then
    run_test "NPU Environment Setup Script" "test -x $HOME/npu-dev/setup_npu_env.sh"
else
    log_warning "NPU environment setup script not found"
fi

# Test example applications (if available)
if [ -f "whisperx_npu_gui_qt6.py" ]; then
    run_test "Whisper NPU Application" "python -m py_compile whisperx_npu_gui_qt6.py"
fi

echo

# 9. System Configuration Check
echo "9. System Configuration"
echo "======================="

run_detailed_test "Kernel Version" "uname -r"
run_detailed_test "Ubuntu Version" "lsb_release -d | cut -f2"
run_detailed_test "Available Memory" "free -h | grep Mem:"
run_detailed_test "CPU Information" "lscpu | grep 'Model name:' | cut -d: -f2 | xargs"

echo

# 10. Recommendations
echo "10. Recommendations"
echo "=================="

# Check kernel version
KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
if [[ $(echo "$KERNEL_VERSION >= 6.14" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
    log_success "Kernel version $KERNEL_VERSION is optimal for NPU"
elif [[ $(echo "$KERNEL_VERSION >= 6.10" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
    log_warning "Kernel version $KERNEL_VERSION is compatible but not optimal (recommend 6.14+)"
else
    log_error "Kernel version $KERNEL_VERSION is too old (minimum 6.10 required)"
fi

# Check memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEMORY_GB" -ge 16 ]; then
    log_success "System has ${MEMORY_GB}GB RAM (optimal for NPU development)"
elif [ "$MEMORY_GB" -ge 8 ]; then
    log_warning "System has ${MEMORY_GB}GB RAM (minimum for NPU development)"
else
    log_error "System has ${MEMORY_GB}GB RAM (insufficient for optimal NPU development)"
fi

echo

# Final Results
echo "=== Verification Summary ==="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    log_success "All tests passed! NPU development environment is ready."
    exit 0
else
    log_error "Some tests failed. Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo
    echo "Troubleshooting suggestions:"
    echo "1. Run the NPU installation script: ./scripts/install_npu_stack.sh"
    echo "2. Check BIOS settings: Enable NPU/IPU in CPU configuration"
    echo "3. Verify kernel version: Ubuntu 25.04+ recommended"
    echo "4. Check driver installation: sudo modprobe amdxdna"
    echo "5. See documentation: NPU-Development/documentation/"
    exit 1
fi