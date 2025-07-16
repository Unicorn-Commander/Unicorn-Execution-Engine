# Gemma 3 27B Custom Execution Engine - Installation Guide

## 🎯 **System Requirements**

### **Hardware Requirements (EXACT MATCH REQUIRED)**
```
CPU: AMD Ryzen AI (Phoenix/Hawk Point/Strix Point)
NPU: AMD NPU Phoenix (16 TOPS, 2GB memory)
iGPU: AMD Radeon 780M (16GB VRAM via HMA)
RAM: 96GB system memory
Storage: 100GB+ free SSD space
Network: High-speed internet for model downloads
```

### **Software Requirements**
```
OS: Ubuntu 25.04+ (Linux kernel 6.14+)
BIOS: NPU enabled, SMART Access Memory enabled
Driver: AMD XDNA drivers installed
```

---

## 🚀 **Quick Installation (Identical Hardware)**

### **One-Command Installation**
```bash
# Download and run automated installer
curl -sSL https://raw.githubusercontent.com/[your-repo]/install.sh | bash

# Or manual installation:
git clone https://github.com/[your-repo]/Unicorn-Execution-Engine.git
cd Unicorn-Execution-Engine
./install_gemma3_system.sh
```

---

## 🔧 **Manual Installation Steps**

### **Step 1: System Prerequisites**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential cmake ninja-build git wget curl
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y pkg-config libssl-dev

# Check hardware compatibility
python3 hardware_checker.py
```

### **Step 2: BIOS Configuration**
```
1. Reboot and enter BIOS (F2/Del during boot)
2. Navigate to Advanced → CPU Configuration
3. Set IPU/NPU → Enabled
4. Set SMART Access Memory → Enabled  
5. Save and Exit
```

### **Step 3: NPU Driver Installation**
```bash
# Install AMD XDNA drivers
wget https://github.com/amd/xdna-driver/releases/latest/download/xrt_202X.X.X_22.04-amd64-xrt.deb
sudo dpkg -i xrt_*.deb
sudo apt-get install -f

# Verify NPU detection
/opt/xilinx/xrt/bin/xrt-smi examine
# Should show: NPU Phoenix detected
```

### **Step 4: iGPU Development Stack**
```bash
# Install ROCm
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_*.deb
sudo amdgpu-install --usecase=dkms,graphics,multimedia,opencl,hip,rocm

# Install Vulkan SDK
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.268-jammy.list https://packages.lunarg.com/vulkan/1.3.268/lunarg-vulkan-1.3.268-jammy.list
sudo apt update
sudo apt install vulkan-sdk
```

### **Step 5: Python Environment**
```bash
# Create virtual environment
python3.11 -m venv gemma3_env
source gemma3_env/bin/activate

# Install base requirements
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers accelerate safetensors
pip install psutil matplotlib seaborn jupyter
```

### **Step 6: MLIR-AIE2 Development Kit**
```bash
# Clone and build MLIR-AIE2
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
export PATH=$PWD/bin:$PATH
```

### **Step 7: VitisAI Installation**
```bash
# Download VitisAI
git clone https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI

# Build Docker container
docker build -t vitis-ai:latest docker/

# Or use pre-built container
docker pull xilinx/vitis-ai-cpu:latest
```

---

## 📦 **Project Setup**

### **Download Unicorn Execution Engine**
```bash
git clone https://github.com/[your-repo]/Unicorn-Execution-Engine.git
cd Unicorn-Execution-Engine

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### **Download Gemma 3 27B Model**
```bash
# Ensure 60GB+ free space
df -h

# Download model (30-60 minutes)
python gemma3_27b_downloader.py

# Verify download
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('google/gemma-2-27b')
print(f'✅ Model ready: {config.num_hidden_layers} layers')
"
```

---

## ✅ **Installation Verification**

### **Hardware Verification**
```bash
# Check NPU
/opt/xilinx/xrt/bin/xrt-smi examine | grep "NPU Phoenix"
# Expected: NPU Phoenix device found

# Check iGPU  
rocm-smi --showuse
# Expected: Radeon 780M detected

# Check memory
free -h | grep Mem
# Expected: ~96GB total memory

# Check Vulkan
vulkaninfo | head -20
# Expected: Vulkan instance created successfully
```

### **Software Verification**
```bash
# Test Python environment
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'ROCm available: {torch.version.hip is not None}')
"

# Test model loading
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('google/gemma-2-27b')
print('✅ Model configuration loaded')
"

# Run hardware checker
python hardware_checker.py
# Expected: All checks passed
```

### **Performance Verification**
```bash
# Run baseline benchmark
python baseline_benchmark.py --model google/gemma-2-27b

# Expected output:
# ✅ Model loaded: 50.3GB, 27B parameters
# 🚀 Average TPS: 5-10 (baseline)
# 💾 Memory usage: ~55GB
# ✅ Ready for Phase 2: Quantization
```

---

## 🐳 **Docker Installation (Alternative)**

### **Build Container**
```bash
# Build development container
docker build -t gemma3-engine:latest .

# Run with hardware access
docker run -it --privileged \
  --device=/dev/accel/accel0 \
  --device=/dev/dri:/dev/dri \
  -v $(pwd):/workspace \
  -v /dev/shm:/dev/shm \
  gemma3-engine:latest
```

### **Container Verification**
```bash
# Inside container
/opt/xilinx/xrt/bin/xrt-smi examine
python hardware_checker.py
python baseline_benchmark.py
```

---

## 🔧 **Troubleshooting**

### **NPU Issues**
```bash
# NPU not detected
sudo modprobe amdxdna
sudo systemctl restart amdxdna

# Check NPU status
dmesg | grep -i npu
lsmod | grep amdxdna

# Reset NPU
sudo /opt/xilinx/xrt/bin/xrt-smi reset --device 0000:c7:00.1
```

### **iGPU Issues**
```bash
# ROCm not working
sudo usermod -a -G render,video $USER
newgrp render

# Vulkan issues
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
vulkaninfo
```

### **Memory Issues**
```bash
# Insufficient memory
sudo sysctl vm.swappiness=10
sudo sysctl vm.vfs_cache_pressure=50

# Check memory allocation
cat /proc/meminfo | grep -E "(MemTotal|MemAvailable|MemFree)"
```

### **Model Download Issues**
```bash
# Download timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Disk space
sudo apt autoremove
sudo apt autoclean
docker system prune -af
```

---

## 📋 **Post-Installation Checklist**

- [ ] NPU Phoenix detected (`xrt-smi examine`)
- [ ] iGPU Radeon 780M available (`rocm-smi`)  
- [ ] 96GB RAM available (`free -h`)
- [ ] Vulkan working (`vulkaninfo`)
- [ ] Python environment ready (`python --version`)
- [ ] PyTorch with ROCm (`torch.version.hip`)
- [ ] Gemma 3 27B downloaded (`50.3GB model`)
- [ ] Baseline benchmark complete (`5-10 TPS`)
- [ ] All verification tests pass (`hardware_checker.py`)

---

## 🎯 **Next Steps**

After successful installation:

1. **Run Baseline:** `python baseline_benchmark.py`
2. **Apply Quantization:** `python gemma3_27b_quantizer.py`  
3. **Test Optimization:** `python gemma3_27b_hybrid_engine.py`
4. **Benchmark Performance:** `python performance_validator.py`

**Target Performance:** 113-162 TPS (vs 5-10 TPS baseline)

---

## 🆘 **Support**

If you encounter issues:
1. Check hardware compatibility: `python hardware_checker.py`
2. Review logs: `journalctl -u amdxdna`
3. Join community: [Discord/GitHub Issues]
4. Report bugs: [GitHub Issues URL]

**Hardware Requirements are STRICT** - this system is optimized for the exact NPU Phoenix + Radeon 780M + 96GB RAM configuration.