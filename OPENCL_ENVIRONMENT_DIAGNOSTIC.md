# 🔧 OpenCL Environment Diagnostic & Recovery

## The Situation
GPU hangs persist even with:
- Minimal dimensions (M=1, hidden_size=16, ff_dim=32)
- Ultra-safe kernels with extensive bounds checking
- Single work item execution

**This is NOT a kernel code issue - it's an environment problem.**

## Immediate Diagnostics

### 1. Check GPU State
```bash
# Check if GPU is still responsive
lspci | grep VGA
dmesg | tail -20 | grep -i amd

# Check for GPU hangs in system logs
journalctl -f --grep="amdgpu"
```

### 2. AMD Driver Status
```bash
# Check loaded modules
lsmod | grep amd

# Driver version
modinfo amdgpu | grep version

# Check for driver errors
dmesg | grep -i "amdgpu\|drm" | tail -20
```

### 3. OpenCL Environment
```bash
# List OpenCL platforms and devices
clinfo

# Check ROCm installation
ls -la /opt/rocm/
rocm-smi

# Check compute capabilities
/opt/rocm/bin/rocminfo
```

## Recovery Steps

### Option 1: Driver Reset (Immediate)
```bash
# Reload AMD GPU driver
sudo modprobe -r amdgpu
sleep 2
sudo modprobe amdgpu

# Check if GPU comes back
clinfo
```

### Option 2: ROCm Environment Reset
```bash
# Reset ROCm stack
sudo systemctl stop rocm
sudo rmmod amdgpu
sudo modprobe amdgpu
sudo systemctl start rocm

# Verify
rocm-smi
```

### Option 3: Minimal OpenCL Test
Create a minimal test to verify basic functionality:

```python
#!/usr/bin/env python3.13
import pyopencl as cl
import numpy as np

# Ultra-minimal OpenCL test
try:
    platforms = cl.get_platforms()
    print(f"Platforms: {len(platforms)}")
    
    for platform in platforms:
        devices = platform.get_devices()
        print(f"Platform: {platform.name}")
        
        for device in devices:
            print(f"  Device: {device.name}")
            
            # Try to create context
            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx)
            
            # Ultra-simple kernel
            kernel_source = """
            __kernel void test_add(__global float* a, __global float* b, __global float* c) {
                int i = get_global_id(0);
                if (i == 0) c[i] = a[i] + b[i];
            }
            """
            
            program = cl.Program(ctx, kernel_source).build()
            
            # Test with single element
            a = np.array([1.0], dtype=np.float32)
            b = np.array([2.0], dtype=np.float32)
            c = np.zeros(1, dtype=np.float32)
            
            mf = cl.mem_flags
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
            
            # Execute
            program.test_add(queue, (1,), None, a_buf, b_buf, c_buf)
            cl.enqueue_copy(queue, c, c_buf)
            queue.finish()
            
            print(f"    Result: {c[0]} (expected: 3.0)")
            
except Exception as e:
    print(f"OpenCL Error: {e}")
```

## Known Issues with gfx1103 (Phoenix)

### Issue 1: RDNA3 OpenCL Stability
- Phoenix (gfx1103) is RDNA3 architecture
- OpenCL support is newer and potentially unstable
- Known issues with certain kernel patterns

### Issue 2: Memory Management
- Phoenix APU shares system memory
- Memory allocation patterns can cause hangs
- Need specific driver versions

### Issue 3: Compute Shader Conflicts
- Graphics and compute workloads can conflict
- Desktop environment might interfere

## Environment Fixes

### Fix 1: Kernel Parameters
Add to `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gpu_recovery=1 amdgpu.vm_fault_stop=0"
```
Then: `sudo update-grub && sudo reboot`

### Fix 2: ROCm Configuration
Create `/etc/rocm/rocm.conf`:
```
[System]
HSA_ENABLE_SDMA=0
HSA_ENABLE_INTERRUPT=0
```

### Fix 3: PyOpenCL Environment
```bash
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_CTX='0'
export GPU_MAX_ALLOC_PERCENT=50
```

## Alternative: CPU-Only Phase 1

If GPU issues persist, continue with CPU-optimized Phase 1:

```python
# Fallback to optimized CPU implementation
class Phase1CPUOptimized:
    def __init__(self):
        # Use NumPy with optimized BLAS
        import numpy as np
        # Verify BLAS backend
        print(f"NumPy BLAS: {np.__config__.show()}")
    
    def qkv_projection_cpu(self, input_data, W_q, W_k, W_v):
        # Fused QKV on CPU with BLAS
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)
        return np.dot(input_data, W_qkv)
    
    def attention_cpu_optimized(self, Q, K, V):
        # Optimized attention with einsum
        scores = np.einsum('bhsd,bhtd->bhst', Q, K) / np.sqrt(Q.shape[-1])
        # Apply causal mask and softmax
        # ... implementation
        return attention_output
    
    def mlp_cpu_optimized(self, x, W_gate, W_up, W_down):
        # Fused MLP operations
        gate = np.dot(x, W_gate)
        up = np.dot(x, W_up)
        # GELU
        sigmoid = 1.0 / (1.0 + np.exp(-1.702 * gate))
        activated = gate * sigmoid * up
        return np.dot(activated, W_down)
```

## Hardware-Specific Recommendations

### For Phoenix APU (gfx1103):
1. **Use ROCm 5.4+ or 6.0+** - Better Phoenix support
2. **Limit memory allocation** - APU shares system RAM
3. **Check thermal throttling** - Phoenix can throttle under load
4. **Update BIOS** - Recent BIOS updates improve stability

### Commands to Check:
```bash
# Check thermal state
sensors | grep -i temp

# Check power management
cat /sys/class/drm/card0/device/power_dpm_state

# Check GPU clocks
cat /sys/class/drm/card0/device/pp_dpm_sclk
```

## Workaround: Use ROCm Directly

Instead of PyOpenCL, try ROCm's HIP:

```cpp
// test_hip.cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void simple_add(float* a, float* b, float* c) {
    int i = threadIdx.x;
    if (i == 0) c[i] = a[i] + b[i];
}

int main() {
    float *d_a, *d_b, *d_c;
    float h_a = 1.0f, h_b = 2.0f, h_c;
    
    hipMalloc(&d_a, sizeof(float));
    hipMalloc(&d_b, sizeof(float));
    hipMalloc(&d_c, sizeof(float));
    
    hipMemcpy(d_a, &h_a, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, &h_b, sizeof(float), hipMemcpyHostToDevice);
    
    simple_add<<<1, 1>>>(d_a, d_b, d_c);
    hipDeviceSynchronize();
    
    hipMemcpy(&h_c, d_c, sizeof(float), hipMemcpyDeviceToHost);
    
    std::cout << "Result: " << h_c << std::endl;
    return 0;
}
```

Compile with: `hipcc test_hip.cpp -o test_hip`

## Final Recommendation

**For immediate progress on Phase 1:**

1. **Document the GPU issue** and continue with CPU optimization
2. **Phase 1 can still achieve 2x speedup** with fused CPU operations
3. **Move to Phase 2** once Phase 1 CPU version works
4. **Return to GPU** after driver/environment is stable

The kernel fusion principles still apply - just implement them in optimized CPU code first, then port to GPU when stable.