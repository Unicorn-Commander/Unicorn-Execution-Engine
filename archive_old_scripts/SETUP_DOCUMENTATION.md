# Unicorn Execution Engine - Setup Documentation

## Environment Setup

### Step 1: Clone Repository ✅
```bash
cd ~/Development
git clone https://github.com/Unicorn-Commander/Unicorn-Execution-Engine
cd Unicorn-Execution-Engine
```

### Step 2: Activate Python Environment ✅
The project uses a pre-configured environment with all ML frameworks:
```bash
source /home/ucadmin/activate-uc1-ai-py311.sh
```

This activates:
- Python 3.11.7
- PyTorch 2.4.0+rocm6.1 (for model loading only, not inference)
- All required dependencies pre-installed

### Step 3: Verify Environment ✅
```bash
python --version  # Should show Python 3.11.7
pip list | grep -E "(torch|transformers|safetensors|fastapi)"
```

Key packages verified:
- ✅ torch 2.7.1
- ✅ transformers 4.52.4
- ✅ safetensors 0.5.3
- ✅ fastapi 0.115.14
- ✅ numpy 2.2.3
- ✅ All other requirements

### Step 4: Download Quantized Model (TODO)
The model needs to be downloaded from HuggingFace:
```bash
# Model location: quantized_models/gemma-3-27b-it-layer-by-layer/
# Source: https://huggingface.co/magicunicorn/gemma-3-27b-npu-quantized
```

### Step 5: Verify Hardware
```bash
# Check NPU
ls -la /dev/accel/accel0
xrt-smi examine

# Check GPU
vulkaninfo --summary | grep "AMD Radeon"
radeontop  # Monitor GPU usage
```

## Architecture Notes

### Pure Hardware System (Port 8006)
- **NO PyTorch/ROCm** during inference
- Uses custom MLIR-AIE2 compiled NPU kernels
- Uses Vulkan compute shaders for GPU
- Direct memory mapping with safetensors

### Key Components
1. **NPU Kernels**: Pre-compiled binaries in `npu_kernels/`
2. **Vulkan Shaders**: GLSL shaders compiled to SPIR-V
3. **Memory Management**: Direct mmap without framework overhead
4. **API Server**: FastAPI server for OpenAI-compatible interface

### Performance Targets
- GPU-only: 8.5 TPS achieved
- With optimizations: 100+ TPS possible
- Target: 81 TPS for production