# 🚀 NPU QUICK START GUIDE

Welcome to the Unicorn Execution Engine with NPU acceleration!

## 📋 Prerequisites

- AMD Phoenix APU with NPU (XDNA1 architecture)
- XRT runtime installed (`/opt/xilinx/xrt/`)
- User in `render` group: `sudo usermod -a -G render $USER`

## 🏃 Quick Start

### 1. Test NPU Functionality
```bash
./test_npu_acceleration.sh
```

### 2. Build with NPU Support
```bash
./build_llama_with_xrt.sh
```

### 3. Run with NPU Acceleration
```bash
# Basic usage with NPU attention
./llama-cli -m model.gguf -p "Hello world" --npu-attention

# With Vulkan GPU acceleration
./llama-cli -m model.gguf -p "Hello world" --gpu-layers 999

# Combined NPU + GPU (maximum performance)
./llama-cli -m model.gguf -p "Hello world" --npu-attention --gpu-layers 999
```

## 🎯 Supported Models

NPU kernels are optimized for:
- **Gemma 3n** - Lightweight variant
- **Gemma 4B** - Standard 4 billion parameter model  
- **Gemma 27B** - Large 27 billion parameter model

Sequence lengths: 128, 256, 512, 1024, 2048 tokens

## 📊 Performance Expectations

| Configuration | Expected Performance |
|--------------|---------------------|
| CPU Only | ~5-10 tok/s |
| Vulkan GPU | ~80-100 tok/s |
| NPU Attention | 200x+ speedup potential |
| NPU + GPU | Maximum performance |

## 🔧 Troubleshooting

### NPU Not Detected
```bash
# Check NPU device
ls -la /dev/accel/accel0

# Verify XRT installation
/opt/xilinx/xrt/bin/xrt-smi examine
```

### Permission Denied
```bash
# Add user to render group
sudo usermod -a -G render $USER
# Log out and back in
```

### XRT Libraries Not Found
```bash
# Set library path
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
```

## 📁 Key Files

- `npu_xrt_compute.cpp` - XRT NPU compute implementation
- `npu_stub.cpp` - NPU integration layer
- `npu_kernels_real/` - Compiled NPU kernels
- `test_npu_acceleration.sh` - Test script

## 🎉 Success Indicators

When NPU is working correctly, you'll see:
```
🧠 NPU ATTENTION FLAG ACTIVE
✅ NPU device opened successfully
✅ NPU AIE Version: 1.1
📋 Selected Gemma3n NPU kernel
✅ NPU ATTENTION COMPLETE
```

## 💡 Tips

1. **Start Small**: Test with smaller models first (2B, 4B)
2. **Monitor Performance**: Use `htop` and GPU monitoring tools
3. **Sequence Length**: NPU performance varies with context size
4. **Memory**: Ensure sufficient system RAM for model loading

## 🦄 Enjoy Your NPU-Accelerated LLM!

The Magic Unicorn NPU acceleration is now at your fingertips. Happy inferencing! ✨