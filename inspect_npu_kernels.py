#!/usr/bin/env python3
import os
import pyxrt

print("🔍 Inspecting Available NPU Kernels")
print("===================================")

os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    # Open NPU device
    device = pyxrt.device(0)
    print("✅ NPU device opened")
    
    # Load validation kernel
    kernel_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin"
    print(f"📦 Loading: {kernel_path}")
    
    xclbin = pyxrt.xclbin(kernel_path)
    uuid = device.register_xclbin(xclbin)
    print(f"✅ XCLBIN registered with UUID: {uuid}")
    
    # Get all available kernels
    kernels = xclbin.get_kernels()
    print(f"\n🔍 Found {len(kernels)} kernel(s):")
    
    for i, kernel_obj in enumerate(kernels):
        name = kernel_obj.get_name()
        print(f"  {i}: {name}")
        
        # Try to create kernel with exact name
        try:
            test_kernel = pyxrt.kernel(device, uuid, name)
            print(f"     ✅ Can create kernel with name '{name}'")
            
            # Get argument info
            try:
                for j in range(10):  # Check up to 10 arguments
                    bank = test_kernel.group_id(j)
                    print(f"     Arg {j}: bank {bank} (0x{bank:X})")
            except:
                pass
                
            break  # Found working kernel
            
        except Exception as e:
            print(f"     ❌ Cannot create kernel '{name}': {e}")
    
    # Also check compute units
    print(f"\n🔧 Checking compute units:")
    cus = xclbin.get_cus()
    for i, cu in enumerate(cus):
        cu_name = cu.get_name()
        print(f"  {i}: {cu_name}")
        
        # Try to create kernel with CU name
        try:
            test_kernel = pyxrt.kernel(device, uuid, cu_name)
            print(f"     ✅ Can create kernel with CU name '{cu_name}'")
        except Exception as e:
            print(f"     ❌ Cannot create with CU name: {e}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()