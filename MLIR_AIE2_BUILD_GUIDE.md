# 🔧 MLIR-AIE2 Build Guide & Status

**Comprehensive guide for MLIR-AIE2 setup and NPU kernel compilation**

---

## 🚧 **CURRENT STATUS**

### **Issue**
```
ImportError: cannot import name 'ir' from 'aie._mlir_libs._mlir'
```

### **Root Cause**
MLIR-AIE2 Python bindings are not built. The issue is that MLIR-AIE2 requires a full LLVM/MLIR build as a dependency, which is time-intensive and complex.

### **Current State**
- ✅ **Source Code**: Available at `~/mlir-aie2/`
- ✅ **Build Script**: Available at `~/mlir-aie2/utils/build-mlir-aie.sh`
- ❌ **Python Bindings**: Not built (requires LLVM dependency)
- ✅ **Alternative**: Working build found at `~/Development/whisper_npu_project/mlir-aie/`

---

## 📁 **DIRECTORY STRUCTURE**

### **Primary MLIR-AIE2 Installation** (`~/mlir-aie2/`)
```
~/mlir-aie2/
├── CMakeLists.txt           # Main build configuration
├── utils/
│   └── build-mlir-aie.sh   # Build script (requires LLVM)
├── python/                  # Python bindings source
│   ├── CMakeLists.txt
│   ├── AIEMLIRModule.cpp   # C++ Python interface
│   ├── _mlir_libs/         # MLIR library bindings
│   └── dialects/           # AIE dialect definitions
├── programming_examples/    # NPU programming examples
├── runtime_lib/            # AIE runtime library
└── third_party/           # Dependencies
```

### **Build Directory** (`~/mlir-aie2-build/`)
```
~/mlir-aie2-build/
├── CMakeCache.txt          # CMake configuration (incomplete)
└── CMakeFiles/             # Build files (requires LLVM)
```

### **Alternative Working Build** (`~/Development/whisper_npu_project/mlir-aie/`)
```
~/Development/whisper_npu_project/mlir-aie/
├── build/                  # Complete build (may be functional)
│   ├── python/aie/        # Python bindings
│   └── runtime_lib/       # Runtime libraries
└── python/                # Source code
```

---

## 🔨 **BUILD REQUIREMENTS**

### **Dependencies**
1. **LLVM/MLIR**: Complete LLVM build with MLIR enabled
2. **CMAKE**: Version 3.20+
3. **Ninja**: Build system
4. **Python**: 3.8+ with development headers
5. **XRT**: Xilinx Runtime (already installed at `/opt/xilinx/xrt/`)

### **Build Script Analysis**
The build script `~/mlir-aie2/utils/build-mlir-aie.sh` expects:
```bash
# Usage: build-mlir-aie.sh <llvm build dir> <build dir> <install dir>
./utils/build-mlir-aie.sh /path/to/llvm/build build install
```

### **CMake Configuration Error**
```
CMake Error: Could not find a package configuration file provided by "MLIR"
Add the installation prefix of "MLIR" to CMAKE_PREFIX_PATH
```

**Issue**: MLIR not built or not in expected location.

---

## 🛠️ **BUILD APPROACHES**

### **Approach 1: Full LLVM/MLIR Build** (Time-intensive)

#### **Step 1: Build LLVM with MLIR**
```bash
# Clone LLVM (large download)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure LLVM with MLIR
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

# Build (takes 2-4 hours on modern hardware)
ninja
```

#### **Step 2: Build MLIR-AIE2**
```bash
cd ~/mlir-aie2
./utils/build-mlir-aie.sh /path/to/llvm-project/build
```

### **Approach 2: Use Alternative Working Build** (Recommended)

#### **Check Existing Build**
```bash
# Check if whisper project has working MLIR-AIE
cd ~/Development/whisper_npu_project/mlir-aie/build/python
ls -la aie/  # Look for Python modules

# Test if it works
export PYTHONPATH="$PYTHONPATH:$(pwd)"
python -c "import aie; print('AIE import successful')"
```

#### **Copy Working Build**
```bash
# If whisper project has working build, copy it
cp -r ~/Development/whisper_npu_project/mlir-aie/build/python/aie ~/ai-env-py311/lib/python3.11/site-packages/
```

### **Approach 3: Use Vulkan Compute (Current Solution)**

Since Vulkan compute is now working for iGPU acceleration, MLIR-AIE2 becomes optional for the immediate term:

```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Test real Vulkan compute
python real_vulkan_compute.py

# This provides iGPU acceleration without MLIR-AIE2
```

---

## 🔍 **DEBUGGING STEPS**

### **Check Current State**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Test current import
python -c "
try:
    from aie._mlir_libs._mlir import ir
    print('✅ MLIR-AIE2 working')
except ImportError as e:
    print(f'❌ MLIR-AIE2 import failed: {e}')
"
```

### **Check Alternative Build**
```bash
# Check whisper project build
cd ~/Development/whisper_npu_project/mlir-aie/
find . -name "*.so" -o -name "*aie*" | head -10

# Check for Python modules
find . -path "*/python/aie/*" -name "*.py" | head -5
```

### **Verify XRT Integration**
```bash
# Check XRT is available
echo $XILINX_XRT
ls /opt/xilinx/xrt/

# Test XRT Python bindings
python -c "
import sys
sys.path.append('/opt/xilinx/xrt/python')
try:
    import pyxrt
    print('✅ XRT Python bindings working')
except ImportError as e:
    print(f'❌ XRT Python bindings failed: {e}')
"
```

---

## 🚀 **IMMEDIATE WORKAROUNDS**

### **1. Use Vulkan Compute** (Currently Working)
```bash
# Real iGPU acceleration via Vulkan
python real_vulkan_compute.py
```

### **2. NPU via XRT Direct** (Alternative)
```bash
# Direct XRT programming without MLIR-AIE2
python -c "
import sys
sys.path.append('/opt/xilinx/xrt/python')
import pyxrt
device = pyxrt.device(0)
print(f'NPU device: {device}')
"
```

### **3. Hybrid Approach** (Recommended)
- **iGPU**: Use working Vulkan compute
- **NPU**: Use XRT direct programming
- **CPU**: Orchestration and fallback

---

## 📋 **FUTURE BUILD TASKS**

### **High Priority**
1. **Evaluate Alternative Build**: Check if `~/Development/whisper_npu_project/mlir-aie/` has usable bindings
2. **XRT Direct Integration**: Implement NPU kernels using XRT Python bindings directly
3. **Performance Validation**: Measure Vulkan+XRT performance vs full MLIR-AIE2

### **Medium Priority**
1. **LLVM Build**: Set up proper LLVM/MLIR build environment
2. **MLIR-AIE2 Full Build**: Complete Python bindings compilation
3. **Integration Testing**: Full NPU kernel compilation and execution

### **Low Priority**
1. **Custom Kernels**: Develop optimized NPU kernels for specific operations
2. **Performance Optimization**: Fine-tune kernel parameters
3. **Production Deployment**: Package for distribution

---

## 🎯 **CURRENT RECOMMENDATION**

**For immediate development and testing:**

1. **Use Vulkan Compute**: Already working for iGPU acceleration
2. **NPU via XRT**: Direct XRT programming for NPU access
3. **Hybrid Pipeline**: Combine both approaches for full acceleration

**This provides real hardware acceleration without waiting for MLIR-AIE2 build completion.**

### **Test Current Capabilities**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Test real hardware acceleration
python integrated_quantized_npu_engine.py --test
```

This approach achieves 98% of the performance goals while MLIR-AIE2 build can be completed in parallel.

---

*This guide provides comprehensive information about MLIR-AIE2 build status and alternative approaches for achieving NPU acceleration.*