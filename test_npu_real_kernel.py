#!/usr/bin/env python3
"""Test real NPU kernel execution using pyxrt"""

import pyxrt
import numpy as np

def test_real_npu():
    print("🧪 Testing Real NPU Kernel Execution")
    print("=" * 50)
    
    try:
        # Open NPU device
        device = pyxrt.device(0)
        print(f"✅ Opened NPU device: {device.get_info(pyxrt.xrt_info_device.name)}")
        
        # Load the validation kernel
        xclbin_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin"
        print(f"\n📦 Loading kernel: {xclbin_path}")
        
        xclbin = pyxrt.xclbin(xclbin_path)
        device.register_xclbin(xclbin)
        uuid = device.get_xclbin_uuid()
        print(f"✅ Kernel loaded with UUID: {uuid}")
        
        # Get kernel handle
        kernel_name = "DPU_PDI_0"
        kernel = pyxrt.kernel(device, uuid, kernel_name, pyxrt.kernel.shared)
        print(f"✅ Got kernel handle: {kernel_name}")
        
        # The validation kernel expects specific buffer sizes
        # Let's create some test buffers
        buffer_size = 1024  # 1KB test buffer
        
        # Allocate buffers with correct flags for NPU
        print(f"\n💾 Allocating NPU buffers...")
        # Use cacheable flag which we know works from previous tests
        bo_in = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, kernel.group_id(0))
        bo_out = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, kernel.group_id(1))
        
        # Fill input buffer with test data
        input_data = np.ones(buffer_size // 4, dtype=np.float32)  # float32 data
        bo_in.write(input_data.tobytes(), 0)  # Write at offset 0
        bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        print("✅ Input buffer filled and synced to NPU")
        
        # Run kernel
        print(f"\n🚀 Executing kernel on NPU...")
        run = kernel(bo_in, bo_out, buffer_size)
        run.wait()
        print("✅ Kernel execution completed!")
        
        # Read results
        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_data = bo_out.read(buffer_size, 0)  # Read from offset 0
        print(f"✅ Output retrieved: {output_data[:10]}...")  # Show first 10 values
        
        print("\n🎉 Real NPU kernel execution successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might mean:")
        print("1. The validation kernel doesn't match our test pattern")
        print("2. We need different buffer configurations")
        print("3. The kernel expects different arguments")
        return False

if __name__ == "__main__":
    test_real_npu()