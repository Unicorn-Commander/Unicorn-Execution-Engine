#!/bin/bash

# NPU Development Stack Installation Script
# Installs all required components for AMD Ryzen AI NPU development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
INSTALL_DIR="$HOME/npu-dev"
PARALLEL_JOBS=$(nproc)

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

log_info "Starting NPU Development Stack Installation"
log_info "Installation directory: $INSTALL_DIR"
log_info "Parallel jobs: $PARALLEL_JOBS"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
    if [[ $(echo "$KERNEL_VERSION >= 6.10" | bc -l) -eq 0 ]]; then
        log_error "Kernel version $KERNEL_VERSION is too old. Need 6.10+"
        exit 1
    fi
    log_success "Kernel version $KERNEL_VERSION is compatible"
    
    # Check for NPU in lspci
    if ! lspci | grep -qi "signal processing"; then
        log_warning "NPU not detected in lspci. Check BIOS settings."
        log_warning "Enable: BIOS → Advanced → CPU Configuration → IPU"
    else
        log_success "NPU device detected"
    fi
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run this script as root"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        python3 \
        python3-pip \
        python3-venv \
        libboost-all-dev \
        libudev-dev \
        libdrm-dev \
        libssl-dev \
        libffi-dev \
        pkg-config \
        dkms \
        bc
    
    log_success "System dependencies installed"
}

# Install XDNA kernel driver
install_xdna_driver() {
    log_info "Installing XDNA kernel driver..."
    
    # Check if driver is already loaded
    if lsmod | grep -q amdxdna; then
        log_success "XDNA driver already loaded"
        return 0
    fi
    
    # Clone and build driver
    if [[ ! -d "xdna-driver" ]]; then
        git clone https://github.com/amd/xdna-driver.git
    fi
    
    cd xdna-driver
    
    # Build driver
    make -C src/driver -j$PARALLEL_JOBS
    
    # Install driver
    sudo make -C src/driver install
    
    # Load driver
    sudo modprobe amdxdna
    
    # Add to modules load list
    echo "amdxdna" | sudo tee /etc/modules-load.d/amdxdna.conf
    
    cd ..
    
    # Verify installation
    if lsmod | grep -q amdxdna; then
        log_success "XDNA driver installed and loaded"
    else
        log_error "XDNA driver installation failed"
        exit 1
    fi
}

# Install XRT (Xilinx Runtime)
install_xrt() {
    log_info "Installing XRT (Xilinx Runtime)..."
    
    # Check if XRT is already installed
    if [[ -f "/opt/xilinx/xrt/setup.sh" ]]; then
        log_success "XRT already installed"
        return 0
    fi
    
    # Clone XRT
    if [[ ! -d "XRT" ]]; then
        git clone https://github.com/Xilinx/XRT.git
    fi
    
    cd XRT
    
    # Install XRT dependencies
    sudo ./src/runtime_src/tools/scripts/xrtdeps.sh
    
    # Build XRT
    cd build
    ./build.sh -j $PARALLEL_JOBS
    
    # Install XRT
    sudo ./build.sh -install
    
    cd ../..
    
    # Verify installation
    source /opt/xilinx/xrt/setup.sh
    if command -v xrt-smi &> /dev/null; then
        log_success "XRT installed successfully"
    else
        log_error "XRT installation failed"
        exit 1
    fi
}

# Install MLIR-AIE (IRON) framework
install_mlir_aie() {
    log_info "Installing MLIR-AIE (IRON) framework..."
    
    # Clone MLIR-AIE
    if [[ ! -d "mlir-aie" ]]; then
        git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
    fi
    
    cd mlir-aie
    
    # Create Python environment
    if [[ ! -d "ironenv" ]]; then
        python3 -m venv ironenv
    fi
    
    source ironenv/bin/activate
    
    # Upgrade pip and install requirements
    pip install --upgrade pip setuptools wheel
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found in mlir-aie, skipping."
    fi
    
    # Build LLVM-AIE (if not already built)
    if [[ ! -d "llvm/build" ]]; then
        log_info "Building LLVM-AIE (this may take a while)..."
        ./utils/clone-llvm.sh
        ./utils/build-llvm.sh
    fi
    
    # Build mlir-aie
    if [[ ! -d "build" ]]; then
        mkdir build
    fi
    
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DAIE_ENABLE_PHOENIX=ON -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm -DAIE_TESTS=OFF
    make -j$PARALLEL_JOBS
    
    # Install Python wheels
    cd ..
    if [[ -d "python_bindings/mlir_aie/dist" ]]; then
        pip install python_bindings/mlir_aie/dist/*.whl
    fi
    
    cd ..
    
    log_success "MLIR-AIE framework installed"
}

# Create Python development environment
create_python_env() {
    log_info "Creating Python development environment..."
    
    # Create main NPU development environment
    if [[ ! -d "npu_dev_env" ]]; then
        python3 -m venv npu_dev_env
    fi
    
    source npu_dev_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install core ML libraries
    pip install \
        onnxruntime>=1.22.0 \
        torch>=2.0.0 \
        transformers>=4.40.0 \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        librosa>=0.10.0 \
        soundfile>=0.12.0 \
        sounddevice>=0.4.0 \
        pydub>=0.25.0 \
        onnx>=1.15.0 \
        netron \
        psutil \
        matplotlib \
        jupyter \
        ipykernel
    
    # Install development tools
    pip install \
        black \
        flake8 \
        pytest \
        mypy \
        pre-commit
    
    log_success "Python development environment created"
}

# Create environment setup script
create_env_script() {
    log_info "Creating environment setup script..."
    
    cat > "$INSTALL_DIR/setup_npu_env.sh" << 'EOF'
#!/bin/bash

# NPU Development Environment Setup Script
# Source this script to set up your NPU development environment

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up NPU Development Environment...${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up XRT environment
if [[ -f "/opt/xilinx/xrt/setup.sh" ]]; then
    source /opt/xilinx/xrt/setup.sh
    echo -e "${GREEN}✓ XRT environment loaded${NC}"
else
    echo "Warning: XRT not found at /opt/xilinx/xrt/setup.sh"
fi

# Set up MLIR-AIE environment
if [[ -d "$SCRIPT_DIR/mlir-aie" ]]; then
    source "$SCRIPT_DIR/mlir-aie/ironenv/bin/activate"
    source "$SCRIPT_DIR/mlir-aie/utils/env_setup.sh"
    echo -e "${GREEN}✓ MLIR-AIE environment loaded${NC}"
else
    echo "Warning: MLIR-AIE not found at $SCRIPT_DIR/mlir-aie"
fi

# Set up Python environment
if [[ -d "$SCRIPT_DIR/npu_dev_env" ]]; then
    source "$SCRIPT_DIR/npu_dev_env/bin/activate"
    echo -e "${GREEN}✓ Python NPU dev environment activated${NC}"
else
    echo "Warning: Python environment not found at $SCRIPT_DIR/npu_dev_env"
fi

# Set environment variables
export NPU_DEV_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo -e "${GREEN}NPU Development Environment Ready!${NC}"
echo "NPU_DEV_ROOT: $NPU_DEV_ROOT"
EOF
    
    chmod +x "$INSTALL_DIR/setup_npu_env.sh"
    log_success "Environment setup script created at $INSTALL_DIR/setup_npu_env.sh"
}

# Verify installation
verify_installation() {
    log_info "Verifying NPU installation..."
    
    # Check XDNA driver
    if lsmod | grep -q amdxdna; then
        log_success "✓ XDNA driver loaded"
    else
        log_error "✗ XDNA driver not loaded"
    fi
    
    # Check XRT
    if command -v xrt-smi &> /dev/null; then
        log_success "✓ XRT installed"
        source /opt/xilinx/xrt/setup.sh
        if xrt-smi examine | grep -q "NPU"; then
            log_success "✓ NPU device detected by XRT"
        else
            log_warning "? NPU device not detected by XRT"
        fi
    else
        log_error "✗ XRT not found"
    fi
    
    # Check MLIR-AIE
    if [[ -d "$INSTALL_DIR/mlir-aie/ironenv" ]]; then
        log_success "✓ MLIR-AIE environment created"
    else
        log_error "✗ MLIR-AIE environment not found"
    fi
    
    # Check Python environment
    if [[ -d "$INSTALL_DIR/npu_dev_env" ]]; then
        log_success "✓ Python development environment created"
    else
        log_error "✗ Python development environment not found"
    fi
    
    # Check device files
    if [[ -e "/dev/accel/accel0" ]]; then
        log_success "✓ NPU device files present"
    else
        log_warning "? NPU device files not found"
    fi
}

# Main installation flow
main() {
    log_info "=== NPU Development Stack Installation ==="
    
    check_prerequisites
    install_system_deps
    install_xdna_driver
    install_xrt
    install_mlir_aie
    create_python_env
    create_env_script
    verify_installation
    
    log_success "=== Installation Complete ==="
    log_info "To set up your development environment, run:"
    log_info "  source $INSTALL_DIR/setup_npu_env.sh"
    log_info ""
    log_info "For usage examples, see: NPU-Development/examples/"
}

# Run main function
main "$@"