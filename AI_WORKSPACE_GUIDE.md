# 🧠 AI WORKSPACE COMPREHENSIVE GUIDE

**Essential guide for working with the AI development environment on UC-1 system**

---

## 🔑 **CRITICAL FIRST STEP**

**ALWAYS activate the AI environment before any AI-related work:**

```bash
source ~/activate-uc1-ai-py311.sh
```

**This is MANDATORY for all AI operations** - without it, frameworks will not be available.

---

## 🗂️ **AI WORKSPACE DIRECTORY STRUCTURE**

### **Primary AI Environment** (`~/`)

#### **Essential Files**
- `~/activate-uc1-ai-py311.sh` - **CRITICAL**: Environment activation script
- `~/ai-env-py311/` - Main Python 3.11 virtual environment
- `~/activate-uc1-ai.sh` - Alternative activation script

#### **NPU Development**
- `~/npu-workspace/` - NPU development workspace
  - `npu_healthcheck.sh` - NPU status verification
  - `npu_monitor.sh` - NPU performance monitoring  
  - `workload_balancer.py` - NPU workload management
  - `Vitis-AI/` - Complete Vitis AI stack

#### **MLIR-AIE2 Infrastructure**
- `~/mlir-aie2/` - MLIR-AIE2 source code
  - `utils/build-mlir-aie.sh` - Build script (requires LLVM)
  - `python/` - Python bindings source
- `~/mlir-aie2-build/` - Build directory (incomplete - needs LLVM)

#### **Models and Data**
- `~/models/` - Model storage directory
- `~/datasets/` - Training/validation datasets
- `~/calibration_data.npy` - Quantization calibration data

### **Development Projects** (`~/Development/github_repos/`)

#### **Primary Project**
- `~/Development/github_repos/Unicorn-Execution-Engine/` - **THIS PROJECT**
  - Main execution engine with real hardware integration
  - Real Vulkan compute implementation
  - NPU+iGPU hybrid execution pipeline

#### **Related AI Projects**
- `~/Development/github_repos/NPU-Development/` - NPU development toolkit
- `~/Development/whisper_npu_project/` - **Contains working MLIR-AIE build**
- `~/Development/kokoro_npu_project/` - TTS with NPU acceleration
- `~/Development/cognitive-companion/` - AI companion project

#### **Infrastructure Projects**
- `~/Development/github_repos/Colonel-Katie/` - AI assistant framework
- `~/Development/github_repos/real-estate-command-center/` - Domain-specific AI

### **System Integration Paths**

#### **NPU Runtime**
- `/opt/xilinx/xrt/` - XRT runtime for NPU
  - `bin/xrt-smi` - NPU management tool
  - `python/` - XRT Python bindings
  - `lib/` - Runtime libraries

#### **AMD GPU Runtime**
- `/opt/rocm/` - AMD ROCm for iGPU/GPU compute
  - `bin/rocm-smi` - GPU management tool
  - `lib/` - HIP/ROCm libraries
  - `include/` - Development headers

---

## 🛠️ **AI ENVIRONMENT DETAILS**

### **Python Environment** (`~/ai-env-py311/`)

#### **Python Installation**
- **Version**: Python 3.11.7
- **Location**: `~/ai-env-py311/bin/python`
- **Activation**: `source ~/activate-uc1-ai-py311.sh`

#### **Pre-installed Frameworks**
```
✅ PyTorch 2.4.0+rocm6.1    - AMD hardware support
✅ TensorFlow 2.19.0        - Google framework
✅ JAX 0.5.0               - Accelerated computing
✅ ONNX Runtime 1.22.0     - Model interoperability
✅ Transformers            - Hugging Face models
✅ Vulkan 1.3.275.1        - Compute shaders
```

#### **Development Tools**
```
✅ Jupyter Lab             - Interactive development
✅ NumPy, SciPy            - Scientific computing
✅ Matplotlib, Plotly      - Visualization
✅ Pandas                  - Data manipulation
✅ Safetensors             - Model serialization
```

### **Hardware Integration**

#### **NPU (Phoenix - 16 TOPS)**
- **Detection**: `xrt-smi examine`
- **Device**: `0000:c7:00.1` 
- **Turbo Mode**: `sudo xrt-smi configure --pmode turbo`
- **Health Check**: `~/npu-workspace/npu_healthcheck.sh`

#### **iGPU (AMD Radeon 780M - RDNA3)**
- **Detection**: `rocm-smi --showuse`
- **Vulkan**: `vulkaninfo --summary`
- **Compute Units**: 12 CUs
- **Target Performance**: 2.7 TFLOPS
- **Memory**: Unified GDDR6

---

## 🚀 **QUICK START WORKFLOWS**

### **1. Environment Verification**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Verify Python and frameworks
python -c "
import torch, tensorflow as tf, jax
print(f'PyTorch: {torch.__version__}')
print(f'TensorFlow: {tf.__version__}') 
print(f'JAX: {jax.__version__}')
print(f'ROCm Available: {torch.cuda.is_available()}')
"

# Check hardware status
xrt-smi examine           # NPU status
rocm-smi --showuse       # iGPU status  
vulkaninfo --summary     # Vulkan support
```

### **2. NPU Development**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Check NPU health
~/npu-workspace/npu_healthcheck.sh

# Monitor NPU performance
~/npu-workspace/npu_monitor.sh

# Test NPU functionality
python ~/test_npu.py
```

### **3. Real Vulkan Compute**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Test real Vulkan compute (NEW!)
cd ~/Development/github_repos/Unicorn-Execution-Engine
python real_vulkan_compute.py
```

### **4. Integrated Engine Testing**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Test integrated engine with real hardware
cd ~/Development/github_repos/Unicorn-Execution-Engine
python integrated_quantized_npu_engine.py --test
```

---

## 🔧 **TROUBLESHOOTING**

### **Environment Issues**

#### **"Command not found" or "Module not found"**
```bash
# SOLUTION: Activate the AI environment
source ~/activate-uc1-ai-py311.sh

# Verify activation worked
which python  # Should show ~/ai-env-py311/bin/python
```

#### **NPU Not Detected**
```bash
# Check NPU driver
sudo modprobe amdxdna
xrt-smi examine

# Verify BIOS setting
# BIOS → Advanced → CPU Configuration → IPU → Enabled
```

#### **iGPU Issues**
```bash
# Check ROCm status
rocm-smi --showuse

# Verify Vulkan support
vulkaninfo --summary

# Test Vulkan device enumeration
python -c "
import vulkan as vk
app_info = vk.VkApplicationInfo(
    sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName='Test',
    apiVersion=vk.VK_API_VERSION_1_0
)
instance_info = vk.VkInstanceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pApplicationInfo=app_info
)
instance = vk.vkCreateInstance(instance_info, None)
devices = vk.vkEnumeratePhysicalDevices(instance)
print(f'Found {len(devices)} Vulkan devices')
"
```

### **Performance Issues**

#### **Low Performance**
```bash
# Enable NPU turbo mode
sudo xrt-smi configure --pmode turbo

# Monitor system temperature
sensors

# Check memory usage
free -h
```

### **MLIR-AIE2 Build Issues**

#### **"No module named 'aie'" Error**
**Root Cause**: MLIR-AIE2 Python bindings not built
**Workaround**: Use Vulkan compute for iGPU acceleration
**Alternative**: Check working build in `~/Development/whisper_npu_project/mlir-aie/`

**To build from source (requires LLVM):**
```bash
# This requires building LLVM/MLIR first (time-intensive)
cd ~/mlir-aie2
./utils/build-mlir-aie.sh <llvm-build-dir>
```

---

## 📊 **PERFORMANCE MONITORING**

### **Real-time Monitoring**
```bash
# NPU monitoring
~/npu-workspace/npu_monitor.sh

# GPU monitoring  
watch -n 1 rocm-smi --showuse

# System monitoring
htop
nvtop  # For GPU monitoring
```

### **Benchmarking**
```bash
# Activate environment
source ~/activate-uc1-ai-py311.sh

# Run comprehensive benchmarks
cd ~/Development/github_repos/Unicorn-Execution-Engine
python validate_performance.py
python hardware_benchmark.py
```

---

## 🎯 **DEVELOPMENT BEST PRACTICES**

### **Always Start With**
```bash
source ~/activate-uc1-ai-py311.sh
```

### **Project Navigation**
```bash
# Primary development directory
cd ~/Development/github_repos/Unicorn-Execution-Engine

# NPU workspace
cd ~/npu-workspace

# Check other AI projects
ls ~/Development/github_repos/
```

### **Hardware Verification Before Development**
```bash
# Quick hardware check
xrt-smi examine && rocm-smi --showuse && vulkaninfo --summary
```

### **Environment Persistence**
```bash
# Add to ~/.bashrc for automatic activation
echo "source ~/activate-uc1-ai-py311.sh" >> ~/.bashrc
```

---

## 🔄 **BACKUP AND MIGRATION**

### **Important Directories to Backup**
- `~/ai-env-py311/` - Complete AI environment
- `~/npu-workspace/` - NPU development workspace
- `~/models/` - Trained models and weights
- `~/Development/github_repos/` - All development projects

### **Environment Recreation**
```bash
# If environment gets corrupted, recreate from backup
cp -r /backup/ai-env-py311 ~/
source ~/activate-uc1-ai-py311.sh
```

---

*This guide provides comprehensive information for working with the AI workspace on the UC-1 system. Always activate the environment first, and refer to this guide for troubleshooting and development workflows.*