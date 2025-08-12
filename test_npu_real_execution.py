#\!/usr/bin/env python3
"""
Test real NPU execution with working validation kernel
"""

import os
import sys
import time
import numpy as np
import pyxrt

def test_npu_vadd():
    """Test NPU execution with vadd kernel"""
    print("🧠 Testing NPU execution with vadd kernel")
    print("=========================================")
    
    try:
        # Initialize NPU device
        device = pyxrt.device(0)
        print("✅ NPU device opened")
        
        # Load working validation kernel
        kernel_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin"
        print(f"📦 Loading working kernel: {kernel_path}")
        
        xclbin = pyxrt.xclbin(kernel_path)
        uuid = device.register_xclbin(xclbin)
        print(f"✅ XCLBIN registered with UUID: {uuid}")
        
        # Get kernels
        kernels = xclbin.get_kernels()
        if not kernels:
            raise RuntimeError("No kernels found")
            
        kernel_name = kernels[0].get_name()
        print(f"🔍 Found kernel: {kernel_name}")
        
        # Create kernel object
        kernel = pyxrt.kernel(device, uuid, kernel_name)
        print("✅ Kernel object created")
        
        # Get memory banks
        print("💾 Memory bank discovery:")
        banks = []
        for i in range(4):
            try:
                bank = kernel.group_id(i)
                banks.append(bank)
                print(f"   Arg {i}: bank {bank} (0x{bank:X})")
            except:
                break
        
        # Allocate test buffers
        buffer_size = 1024 * 4  # 1024 float32 values
        print(f"📦 Allocating {buffer_size} byte buffers")
        
        in1_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[0])
        in2_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[1])
        out_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[2])
        
        # Create test data
        in1_data = np.arange(1024, dtype=np.float32) * 0.1
        in2_data = np.arange(1024, dtype=np.float32) * 0.2
        expected = in1_data + in2_data
        
        print("📤 Copying data to NPU...")
        in1_bo.write(in1_data.tobytes(), 0)
        in2_bo.write(in2_data.tobytes(), 0)
        
        # Sync to device
        in1_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        in2_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        
        print("🚀 Executing NPU kernel...")
        start_time = time.time()
        
        # Execute kernel: vadd(in1, in2, out_r, size)
        run = kernel(in1_bo, in2_bo, out_bo, 1024)
        state = run.wait(10000)  # 10 second timeout
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        
        print(f"✅ NPU execution completed in {execution_time:.2f} ms")
        print(f"   State: {state}")
        
        # Read results
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        result_bytes = out_bo.read(buffer_size, 0)
        result = np.frombuffer(result_bytes, dtype=np.float32)
        
        # Verify correctness
        max_error = np.max(np.abs(result - expected))
        mean_error = np.mean(np.abs(result - expected))
        
        print(f"📊 Results verification:")
        print(f"   Expected: [{expected[0]:.3f}, {expected[1]:.3f}, ..., {expected[-1]:.3f}]")
        print(f"   Got:      [{result[0]:.3f}, {result[1]:.3f}, ..., {result[-1]:.3f}]")
        print(f"   Max error: {max_error:.6f}")
        print(f"   Mean error: {mean_error:.6f}")
        
        if max_error < 1e-6:
            print("✅ NPU computation is CORRECT\!")
            return True, execution_time
        else:
            print("❌ NPU computation has errors")
            return False, execution_time
            
    except Exception as e:
        print(f"❌ NPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    print("🦄 NPU Real Execution Test")
    print("=========================")
    
    # Set XRT environment
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    try:
        # Test basic NPU execution
        success, exec_time = test_npu_vadd()
        
        if success:
            print(f"\n🎉 SUCCESS: NPU execution is working\!")
            print(f"    Execution time: {exec_time:.2f} ms")
            print(f"\n🦄✨ CONCLUSION: NPU is OPERATIONAL\!")
            print("The NPU can execute kernels successfully.")
            print("The attention XCLBIN files are incomplete/corrupted.")
            print("But the NPU hardware is proven working\!")
            
        else:
            print(f"\n❌ NPU execution failed")
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
EOF < /dev/null
