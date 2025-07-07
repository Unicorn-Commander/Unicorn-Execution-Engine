#!/bin/bash

# Simplified NPU Environment Verification Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_info "=== Essential NPU Environment Verification ==="

# 1. Verify XRT


# 2. Verify Python Environment
log_info "\n2. Verifying Python Environment..."
if python3 -c "import onnxruntime; print(onnxruntime.__version__)" &> /dev/null; then
    log_success "  [✓] ONNX Runtime is installed"
else
    log_error "  [✗] ONNX Runtime is not installed"
    exit 1
fi

log_success "\n=== Essential Verification Complete ==="