#!/bin/bash

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

log_info "Starting MLIR-AIE Build"
log_info "Installation directory: $INSTALL_DIR"
log_info "Parallel jobs: $PARALLEL_JOBS"

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
        touch requirements.txt
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
    cmake .. -DCMAKE_BUILD_TYPE=Release -DAIE_ENABLE_PHOENIX=ON -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm -DAIE_ENABLE_PYTHON_BINDINGS=ON -DAIE_TESTS=OFF
    make -j$PARALLEL_JOBS
    
    cd ..
    
    log_success "MLIR-AIE framework installed"
}

# Main installation flow
main() {
    log_info "=== MLIR-AIE Build ==="
    
    install_mlir_aie
    
    log_success "=== Installation Complete ==="
}

# Run main function
main "$@"
