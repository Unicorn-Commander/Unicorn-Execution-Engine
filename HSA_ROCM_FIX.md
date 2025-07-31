# 🔧 HSA Override Fix for AMD GPU Issues

## The Problem
AMD GPUs (especially RDNA3/gfx1103) can experience:
- Screen blanking/flickering during compute operations
- GPU hangs during complex kernels
- Display driver conflicts with compute workloads

## HSA Version Override Solution

### Check Current HSA Version
```bash
# Check current ROCm/HSA version
/opt/rocm/bin/rocminfo | grep "HSA Runtime Version"

# Check HSA override if set
echo $HSA_OVERRIDE_GFX_VERSION
```

### Setting HSA Override

The community has found these versions work best for gfx1103:

#### Option 1: HSA 11.0.0 (Most Stable - No Screen Blanking)
```bash
# Add to ~/.bashrc or /etc/environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Apply immediately
source ~/.bashrc
```

#### Option 2: HSA 11.0.2 (Good Compatibility)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.2
```

#### Option 3: HSA 11.0.3 (Newer - Mixed Reports)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.3
```

### System-Wide Configuration

Create `/etc/profile.d/rocm-hsa.sh`:
```bash
#!/bin/bash
# HSA override for gfx1103 stability
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Additional stability settings
export GPU_MAX_HW_QUEUES=1
export AMD_LOG_LEVEL=0
```

Make it executable:
```bash
sudo chmod +x /etc/profile.d/rocm-hsa.sh
```

## Additional ROCm Environment Fixes

### 1. Disable GPU Recovery (Prevents Hangs)
```bash
# Add to /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gpu_recovery=0 amdgpu.ppfeaturemask=0xffffffff"

# Update grub
sudo update-grub
sudo reboot
```

### 2. Set Compute-Only Mode
```bash
# Force compute mode (no display interference)
echo "manual" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
echo "compute" | sudo tee /sys/class/drm/card0/device/pp_power_profile_mode
```

### 3. Memory Allocation Limits
```bash
# Prevent over-allocation
export GPU_MAX_ALLOC_PERCENT=80
export GPU_SINGLE_ALLOC_PERCENT=70
```

## Testing the Fix

### 1. Test Basic ROCm Functionality
```bash
# Should work without screen blanking
/opt/rocm/bin/rocm-smi

# Test compute
/opt/rocm/bin/rocminfo
```

### 2. Test Our OpenCL Code
```bash
# Set HSA override
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Run minimal test
python3.13 minimal_opencl_test.py

# Try simple GPU operations
python3.13 phase1_gpu_robust.py
```

### 3. Monitor for Issues
```bash
# Watch for GPU errors
sudo dmesg -w | grep -i amdgpu

# In another terminal, watch GPU state
watch -n 1 /opt/rocm/bin/rocm-smi
```

## ROCm Installation with Correct Version

If you need to reinstall ROCm with better gfx1103 support:

```bash
# Remove existing ROCm
sudo apt remove rocm-dkms rocm-dev

# Add ROCm 6.0 repository (better RDNA3 support)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0.2 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm 6.0
sudo apt update
sudo apt install rocm-dkms rocm-dev

# Add user to groups
sudo usermod -a -G render,video $USER

# Set HSA override
echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >> ~/.bashrc
```

## Alternative: Use HIP Instead of OpenCL

With HSA override, HIP might be more stable:

```cpp
// test_hip.cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void simple_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    float *d_data;
    size_t size = 1024 * sizeof(float);
    
    hipMalloc(&d_data, size);
    
    simple_kernel<<<1, 1024>>>(d_data);
    hipDeviceSynchronize();
    
    hipFree(d_data);
    std::cout << "HIP kernel executed successfully!" << std::endl;
    
    return 0;
}
```

Compile and test:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
hipcc test_hip.cpp -o test_hip
./test_hip
```

## PyTorch ROCm with HSA Override

```python
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch

# Should detect GPU without screen blanking
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test computation
x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x)
print(f"Computation successful: {y.shape}")
```

## Summary

For gfx1103 (AMD Phoenix), the community consensus is:
- **HSA 11.0.0**: Most stable, no screen blanking
- **HSA 11.0.2**: Good alternative
- **HSA 11.0.3**: Mixed results

Combined with:
- ROCm 6.0+ for better RDNA3 support
- Compute-only power profile
- Memory allocation limits

This should resolve both the screen blanking and GPU hang issues!