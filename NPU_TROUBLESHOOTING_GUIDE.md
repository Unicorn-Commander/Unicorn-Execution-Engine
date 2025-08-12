# NPU Troubleshooting Guide

*Common issues and solutions for NPU integration with llama.cpp*

## 🚨 Quick Diagnostics

Run this first to check your NPU setup:
```bash
python3 test_gemma_npu_integration.py
```

Expected output:
```
✅ NPU device opened successfully
✅ 5 Gemma kernels available
✅ llama.cpp NPU flag found
✅ All tests passed!
```

## 🔧 Common Issues and Solutions

### 1. NPU Device Not Found

**Error:**
```
Failed to open NPU device /dev/accel/accel0
Permission denied
```

**Solutions:**

#### Check device exists:
```bash
ls -la /dev/accel/
# Should show: crw-rw---- 1 root render ... accel0
```

#### Add user to render group:
```bash
sudo usermod -a -G render $USER
# IMPORTANT: Logout and login again for changes to take effect
```

#### Verify group membership:
```bash
groups | grep render
# Should show: ... render ...
```

#### Check driver loaded:
```bash
lsmod | grep amdxdna
# Should show: amdxdna module loaded
```

### 2. NPU Buffer Creation Failed

**Error:**
```
Failed to create NPU buffer objects
ioctl error: Invalid argument
```

**Solutions:**

#### Check dmesg for details:
```bash
sudo dmesg | tail -50 | grep amdxdna
```

#### Common causes:
- Wrong memory bank IDs (must use 131071, 65536, 65537)
- Buffer size not 4KB aligned
- NPU already in use by another process

#### Kill any hanging NPU processes:
```bash
# Find NPU processes
lsof /dev/accel/accel0

# Kill if needed
kill -9 <PID>
```

### 3. Kernel File Not Found

**Error:**
```
Loading NPU kernel: ../npu_kernels_compiled/gemma3_4b_attention.xclbin
NPU kernel not found
```

**Solutions:**

#### Check kernel files exist:
```bash
ls -la npu_kernels_compiled/*.xclbin
```

#### Run from correct directory:
```bash
cd /home/ucadmin/Development/Unicorn-Execution-Engine
./llama.cpp/build/bin/llama-cli ...
```

#### Use absolute paths if needed:
```bash
export NPU_KERNEL_PATH=/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_compiled
```

### 4. Matrix Multiplication Assertion

**Error:**
```
GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
```

**Solutions:**

This is a model compatibility issue, not NPU-related:
- Use a proper Gemma model in GGUF format
- TinyLlama has incompatible dimensions
- Convert model with correct architecture

### 5. NPU Flag Not Recognized

**Error:**
```
Unknown option: --npu-attention
```

**Solutions:**

#### Rebuild with NPU support:
```bash
cd llama.cpp
rm -rf build
cmake -B build -DGGML_VULKAN=ON -DGGML_NPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

#### Verify NPU support compiled in:
```bash
./build/bin/llama-cli --help | grep npu
# Should show: --npu-attention option
```

### 6. Performance Issues

**Symptom:** NPU not providing expected speedup

**Diagnostics:**

#### Check NPU is actually being used:
```bash
# Run with verbose output
./llama.cpp/build/bin/llama-cli ... 2>&1 | grep NPU
```

Should see:
```
🧠 NPU ATTENTION CALLED - Using Direct Runtime
✅ NPU device opened successfully
⚡ NPU HARDWARE EXECUTION
```

#### Monitor NPU usage:
```bash
# In another terminal while running
sudo dmesg -w | grep amdxdna
```

#### Check kernel selection:
- Gemma 4B should use: `gemma3_4b_attention.xclbin`
- Gemma 27B should use: `gemma3_27b_attention.xclbin`

### 7. Memory Errors

**Error:**
```
Failed to map NPU buffer
Cannot allocate memory
```

**Solutions:**

#### Check available memory:
```bash
free -h
```

#### Clear NPU memory:
```bash
# Reboot is most reliable
sudo reboot

# Or try module reload
sudo rmmod amdxdna
sudo modprobe amdxdna
```

## 🔍 Advanced Debugging

### Enable NPU Debug Output

Add to your environment:
```bash
export NPU_DEBUG=1
export AMDXDNA_DEBUG=1
```

### Trace NPU Operations

```bash
# Trace all ioctl calls
sudo strace -e ioctl ./llama.cpp/build/bin/llama-cli ... 2>&1 | grep accel
```

### Check NPU Hardware Info

```bash
# Get detailed NPU info
sudo cat /sys/class/accel/accel0/device/aie_version
sudo cat /sys/class/accel/accel0/device/aie_metadata
```

### Test NPU Directly

```bash
# Compile and run hardware test
g++ -o test_npu test_real_npu_integration.cpp -std=c++17
./test_npu
```

## 📋 Verification Checklist

Before reporting issues, verify:

- [ ] User is in `render` group (logout/login after adding)
- [ ] `/dev/accel/accel0` exists with correct permissions
- [ ] `amdxdna` kernel module is loaded
- [ ] NPU kernel files exist in `npu_kernels_compiled/`
- [ ] Running from correct directory
- [ ] llama.cpp built with `-DGGML_NPU=ON`
- [ ] Using compatible model (Gemma, not TinyLlama)
- [ ] No other processes using NPU

## 🆘 Getting Help

If issues persist:

1. **Collect diagnostics:**
   ```bash
   python3 test_gemma_npu_integration.py > npu_diagnostics.txt 2>&1
   dmesg | grep -E "(amdxdna|accel)" >> npu_diagnostics.txt
   lsmod | grep amdxdna >> npu_diagnostics.txt
   ls -la /dev/accel/ >> npu_diagnostics.txt
   ```

2. **Check versions:**
   ```bash
   uname -r  # Kernel version (should be 6.14+)
   # NPU driver version from dmesg
   ```

3. **Reference working configuration:**
   - OS: Ubuntu with kernel 6.14+
   - NPU: AMD Phoenix (XDNA1)
   - Driver: amdxdna native kernel driver
   - Runtime: Direct IOCTL (no XRT required)

## ✅ Success Indicators

When everything is working correctly:

1. **Device check:** `/dev/accel/accel0` accessible
2. **Kernel check:** 5 `.xclbin` files in `npu_kernels_compiled/`
3. **Build check:** `--npu-attention` flag available
4. **Runtime check:** NPU messages appear during inference
5. **Performance check:** Significant speedup vs CPU-only

Remember: The NPU provides massive acceleration for attention operations. If you don't see speedup, verify the NPU is actually being used!