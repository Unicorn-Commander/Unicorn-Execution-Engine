# NPU Development Software Requirements

## Overview
This document details all software components required for AMD Ryzen AI NPU development, including gaps we encountered and solutions implemented during the Whisper NPU project.

## Hardware Prerequisites

### Supported NPU Architectures
- **AMD Ryzen AI Phoenix** (Verified: NPU with firmware 1.5.5.391)
- **AMD Ryzen AI Hawk Point** (Compatible)
- **AMD Ryzen AI Strix** (Compatible)

### BIOS Configuration
```
BIOS → Advanced → CPU Configuration → IPU → Enabled
```
**Note**: Secure Boot can remain enabled on Ubuntu 25.04+ (signed NPU drivers available)

## Operating System Requirements

### Recommended OS
- **Ubuntu 25.04+** (Native amdxdna driver support)
- **Linux Kernel 6.14+** (includes amdxdna.ko)
- **Alternative**: Ubuntu 24.04+ with HWE stack (kernel 6.11+)

### Desktop Environment
- **KDE6** (Recommended for Qt6 compatibility)
- **GNOME** (Compatible)

## Core NPU Software Stack

### 1. XDNA Kernel Driver (CRITICAL GAP)
**Status**: Not included in standard distributions before Ubuntu 25.04

**Manual Installation Required**:
```bash
# Clone driver source
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver

# Build and install
make -C src/driver
sudo make -C src/driver install
sudo modprobe amdxdna
```

**Verification**:
```bash
lsmod | grep amdxdna  # Should show loaded driver
lspci | grep -i "signal processing"  # Should show NPU device
ls /dev/accel/  # Should show accel devices
```

### 2. XRT (Xilinx Runtime) (MAJOR GAP)
**Status**: Not available in package managers, must build from source

**Installation**:
```bash
# Dependencies
sudo apt install -y cmake build-essential libboost-all-dev
sudo apt install -y libudev-dev libdrm-dev

# Clone and build XRT
git clone https://github.com/Xilinx/XRT.git
cd XRT
./src/runtime_src/tools/scripts/xrtdeps.sh
cd build
./build.sh

# Install
sudo ./build.sh -install
```

**Environment Setup**:
```bash
source /opt/xilinx/xrt/setup.sh
```

**Verification**:
```bash
xrt-smi examine  # Should show NPU Phoenix device
```

### 3. MLIR-AIE (IRON) Framework (CRITICAL GAP)
**Status**: Complex build process, multiple dependencies

**Installation**:
```bash
# Clone repository
git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Create Python environment
python -m venv ironenv
source ironenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build LLVM-AIE (Peano)
./utils/clone-llvm.sh
./utils/build-llvm.sh

# Build mlir-aie
mkdir build && cd build
cmake .. -DLLVM_DIR=../llvm/build/lib/cmake/llvm
make -j$(nproc)

# Install Python wheels
pip install ../python_bindings/mlir_aie/dist/*.whl
```

**Environment Setup**:
```bash
source mlir-aie/ironenv/bin/activate
source mlir-aie/utils/env_setup.sh
```

## Python Environment Dependencies

### Core ML Libraries
```bash
pip install \
    onnxruntime>=1.22.0 \
    torch>=2.0.0 \
    transformers>=4.40.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0
```

### Audio Processing
```bash
pip install \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    sounddevice>=0.4.0 \
    pydub>=0.25.0
```

### NPU-Specific (When Available)
```bash
# VitisAI ExecutionProvider (if available)
pip install vitis-ai-runtime

# ROCm for AMD GPU fallback
pip install torch-rocm
```

## Development Tools

### Debugging and Profiling
```bash
# AMD ROCm tools
sudo apt install rocm-dev rocm-utils

# Performance monitoring
sudo apt install htop nvtop
pip install psutil gpustat
```

### ONNX Model Tools
```bash
pip install \
    onnx>=1.15.0 \
    onnx-tools \
    netron  # Model visualization
```

## Software Gaps Encountered

### 1. Missing NPU Execution Providers
**Problem**: Standard ONNX Runtime lacks NPU execution providers
**Solution**: Custom XRT integration with manual NPU scheduling

### 2. Limited NPU Memory Management
**Problem**: No high-level NPU memory allocation APIs
**Solution**: Temporary file-based memory management system

### 3. No NPU Model Optimization Tools
**Problem**: No tools to optimize ONNX models specifically for NPU
**Solution**: Custom graph optimization and batching strategies

### 4. Incomplete Documentation
**Problem**: Sparse documentation for NPU programming
**Solution**: Reverse engineering and extensive testing

## Installation Verification Script

See: `NPU-Development/scripts/verify_npu_setup.sh`

## Known Issues and Workarounds

### XRT Build Failures
**Issue**: Missing dependencies or version conflicts
**Workaround**: Use Docker container with pre-built XRT

### MLIR-AIE Python Import Errors
**Issue**: Complex dependency chain
**Workaround**: Use wheel packages when available

### NPU Device Not Detected
**Issue**: Driver not loaded or BIOS settings
**Workaround**: Manual driver reload and BIOS verification

## Future Software Needs

### For Computer Vision
- OpenCV with NPU backend
- TensorRT-like optimization for NPU
- Custom convolution kernels

### For LLM Inference
- Quantization tools for NPU
- KV-cache management
- Multi-token generation optimization

### For Embeddings
- Vector database integration
- Batch processing optimization
- Memory-efficient similarity search

## Resources

- **AMD NPU Documentation**: Limited, mostly hardware specs
- **Vitis AI**: Xilinx/AMD AI development platform
- **MLIR-AIE**: Low-level NPU programming framework
- **XRT**: Runtime for Xilinx/AMD accelerators