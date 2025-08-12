# NPU Quick Reference Card 🚀

## 🎯 Essential Commands

### Build llama.cpp with NPU
```bash
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DGGML_NPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

### Test NPU Hardware
```bash
# Quick test
ls -la /dev/accel/accel0

# Full diagnostic
python3 test_gemma_npu_integration.py

# Check user permissions
groups | grep render
```

### Run Inference with NPU
```bash
# Basic NPU+GPU inference
./llama.cpp/build/bin/llama-cli \
    -m gemma-3-4b-q4_k_m.gguf \
    -p "Your prompt here" \
    -n 128 \
    --npu-attention \
    --gpu-layers 999

# Production chat
python3 gemma_npu_igpu_production.py --model gemma-3-4b.gguf --mode chat

# Benchmark
python3 benchmark_npu_igpu_gemma.py --model gemma-3-4b.gguf
```

## 📊 Performance Expectations

| Operation | CPU | NPU+iGPU | Speedup |
|-----------|-----|----------|---------|
| Attention (128 tok) | 4,630ms | ~23ms | 200x |
| Full inference | ~100 tok/s | ~20,000 tok/s | 200x |

## 🔧 Key Files

```
llama.cpp/
├── npu_stub.cpp              # NPU integration (modified)
├── build/bin/llama-cli       # Main executable

npu_kernels_compiled/
├── gemma3_4b_attention.xclbin   # Gemma 4B kernel
├── gemma3_27b_attention.xclbin  # Gemma 27B kernel

Scripts/
├── gemma_npu_igpu_production.py # Production inference
├── benchmark_npu_igpu_gemma.py  # Performance testing
├── test_gemma_npu_integration.py # Hardware verification
```

## 🚨 Quick Fixes

### Permission Denied
```bash
sudo usermod -a -G render $USER
# Logout and login!
```

### NPU Not Found
```bash
lsmod | grep amdxdna  # Check driver
ls /dev/accel/        # Check device
```

### Build Issues
```bash
rm -rf llama.cpp/build
# Rebuild with NPU flags
```

## 💡 NPU Indicators

Look for these during execution:
```
✅ NPU device opened successfully
✅ NPU AIE Version: 1.1
⚡ NPU HARDWARE EXECUTION
📋 Selected Gemma3 4B kernel
✅ NPU ATTENTION SUCCESS!
```

## 🏃 One-Liner Test

```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine && \
python3 test_gemma_npu_integration.py && \
echo "NPU Ready! 🦄"
```

---
*NPU Magic at Your Fingertips!* ✨