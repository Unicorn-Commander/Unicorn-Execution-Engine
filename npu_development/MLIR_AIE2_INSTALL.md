# MLIR-AIE2 Installation Guide

## Option 1: Pre-built Installation
```bash
# Download pre-built MLIR-AIE2 (if available)
wget https://github.com/Xilinx/mlir-aie/releases/latest/download/mlir-aie-linux-x64.tar.gz
tar -xzf mlir-aie-linux-x64.tar.gz -C /opt/xilinx/
export PATH="/opt/xilinx/mlir-aie/bin:$PATH"
```

## Option 2: Build from Source
```bash
# Clone MLIR-AIE2 repository
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Install dependencies
sudo apt-get install cmake ninja-build clang lld

# Build MLIR-AIE2
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;host" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

ninja
sudo ninja install
```

## Option 3: VitisAI Integration
```bash
# Use VitisAI which includes MLIR-AIE2
docker pull xilinx/vitis-ai-cpu:latest
# Or install VitisAI locally with MLIR-AIE2 support
```

## Verification
```bash
# Test MLIR-AIE2 installation
aie-opt --version
aie-translate --version
```
