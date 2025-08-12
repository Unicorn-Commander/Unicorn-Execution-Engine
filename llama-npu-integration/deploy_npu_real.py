#!/usr/bin/env python3
"""
Deploy NPU Backend on Real Hardware
Uses PyXRT to execute compiled kernels
"""

import os
import sys
import numpy as np
import time
import pyxrt

def test_npu_deployment():
    """Test NPU deployment with real hardware"""
    print("🦄 Deploying NPU Backend on Real Hardware")
    print("=========================================\n")
    
    # Initialize NPU device
    print("1. Initializing NPU device...")
    try:
        device = pyxrt.device(0)
        device_name = device.get_info(pyxrt.info.device.name)
        print(f"✅ NPU Device: {device_name}")
    except Exception as e:
        print(f"❌ Failed to open NPU device: {e}")
        return False
    
    # Check for kernel files
    print("\n2. Checking kernel files...")
    kernel_files = {
        128: "attention_gemma3_4b_128.xclbin",
        256: "attention_gemma3_4b_256.xclbin", 
        512: "attention_gemma3_4b_512.xclbin",
        1024: "attention_gemma3_4b_1024.xclbin"
    }
    
    available_kernels = {}
    for seq_len, filename in kernel_files.items():
        build_path = f"build/{filename}"
        if os.path.exists(build_path):
            print(f"✅ Found kernel for seq_len={seq_len}: {filename}")
            available_kernels[seq_len] = build_path
        else:
            print(f"❌ Missing kernel: {filename}")
    
    if not available_kernels:
        print("❌ No kernels found!")
        return False
    
    # Test loading a kernel
    print("\n3. Loading NPU kernel...")
    test_seq_len = 256
    if test_seq_len not in available_kernels:
        test_seq_len = list(available_kernels.keys())[0]
    
    xclbin_path = available_kernels[test_seq_len]
    print(f"Loading: {xclbin_path}")
    
    try:
        xclbin = pyxrt.xclbin(xclbin_path)
        device.register_xclbin(xclbin)
        print("✅ XCLBIN loaded successfully!")
    except Exception as e:
        print(f"⚠️  Failed to load XCLBIN: {e}")
        print("This might be expected if kernel format doesn't match device")
        return False
    
    # Try to create kernel object
    print("\n4. Creating kernel object...")
    kernel_names = ["DPU_PDI_0", "attention", "attention_256", "mm2s", "s2mm"]
    
    kernel = None
    for kname in kernel_names:
        try:
            kernel = pyxrt.kernel(device, xclbin.get_uuid(), kname)
            print(f"✅ Created kernel: {kname}")
            break
        except Exception as e:
            print(f"   Tried {kname}: {e}")
    
    if not kernel:
        print("❌ Could not create kernel object")
        return False
    
    # Allocate buffers
    print("\n5. Allocating NPU buffers...")
    batch = 1
    heads = 16
    seq_len = test_seq_len
    head_dim = 64
    tensor_size = batch * heads * seq_len * head_dim
    
    try:
        # Use correct memory banks for Phoenix NPU
        q_bo = pyxrt.bo(device, tensor_size * 4, pyxrt.bo.flags.cacheable, 131071)
        k_bo = pyxrt.bo(device, tensor_size * 4, pyxrt.bo.flags.cacheable, 131071)
        v_bo = pyxrt.bo(device, tensor_size * 4, pyxrt.bo.flags.cacheable, 131071)
        out_bo = pyxrt.bo(device, tensor_size * 4, pyxrt.bo.flags.cacheable, 131071)
        
        print(f"✅ Allocated {4 * tensor_size * 4 / 1024 / 1024:.2f} MB of NPU memory")
    except Exception as e:
        print(f"❌ Failed to allocate buffers: {e}")
        return False
    
    # Initialize test data
    print("\n6. Running NPU kernel...")
    q_data = np.random.randn(tensor_size).astype(np.float32) * 0.1
    k_data = np.random.randn(tensor_size).astype(np.float32) * 0.1
    v_data = np.random.randn(tensor_size).astype(np.float32) * 0.1
    
    # Copy to device
    q_bo.write(q_data.tobytes())
    k_bo.write(k_data.tobytes())
    v_bo.write(v_data.tobytes())
    
    q_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    k_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    v_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    # Execute kernel
    try:
        run = kernel(q_bo, k_bo, v_bo, out_bo, batch, heads, seq_len, head_dim, 1)
        
        start_time = time.time()
        run.wait()
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000
        print(f"✅ NPU kernel executed in {execution_time:.2f} ms")
        
        # Read results
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output = np.frombuffer(out_bo.read(tensor_size * 4), dtype=np.float32)
        
        print(f"✅ Output shape: {output.shape}, mean: {output.mean():.4f}, std: {output.std():.4f}")
        
        # Calculate performance
        flops = 2 * batch * heads * seq_len * seq_len * head_dim
        tflops = (flops / 1e12) / (execution_time / 1000)
        print(f"✅ Performance: {tflops:.2f} TFLOPS")
        
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        return False
    
    print("\n🎉 NPU deployment successful!")
    return True

def integrate_with_cpp():
    """Show how to integrate with C++ code"""
    print("\n\n📝 C++ Integration Guide")
    print("========================\n")
    
    cpp_code = """
// To use this NPU backend in your C++ code:

#include <dlfcn.h>
#include <pyxrt.h>  // If available

// Or use dlopen to load XRT dynamically:
void* xrt_lib = dlopen("libxrt_core.so", RTLD_LAZY);
if (xrt_lib) {
    // Load function pointers
    auto device_open = (xrtDeviceHandle(*)(unsigned int))dlsym(xrt_lib, "xrtDeviceOpen");
    auto bo_alloc = (xrtBufferHandle(*)(xrtDeviceHandle, size_t, int))dlsym(xrt_lib, "xrtBOAlloc");
    
    // Use the functions
    auto device = device_open(0);
    auto buffer = bo_alloc(device, size, XRT_BO_FLAGS_CACHEABLE);
}

// Build with:
// g++ -o test test.cpp -ldl -lxrt_core -L/opt/xilinx/xrt/lib
"""
    print(cpp_code)

if __name__ == "__main__":
    # Change to build directory if needed
    if os.path.exists("build"):
        os.chdir("build")
    
    success = test_npu_deployment()
    
    if success:
        integrate_with_cpp()
        print("\n✅ NPU is ready for llama.cpp integration!")
        print("\nNext steps:")
        print("1. Rebuild llama-npu-integration with XRT support")
        print("2. Link with -lxrt_core -lxrt_coreutil")
        print("3. Run llama.cpp with --npu-attention flag")
    else:
        print("\n⚠️  NPU deployment needs attention")
        print("Check:")
        print("- XRT version compatibility")
        print("- Kernel format (might need recompilation)")
        print("- Driver flags: aie2_control_flags=7")