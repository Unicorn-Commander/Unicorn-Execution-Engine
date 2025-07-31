#!/usr/bin/env python3.13
"""
Test NPU kernel access patterns
"""

import pyxrt
import os

print("🔍 NPU Kernel Discovery")
print("=" * 50)

# Open device
device = pyxrt.device(0)
print("✅ Device opened")

# Try different XCLBINs
xclbins = [
    "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin",
    "/opt/xilinx/xrt/amdxdna/bins/17f0_11/validate.xclbin",
    "/opt/xilinx/xrt/amdxdna/bins/17f0_10/validate.xclbin",
]

for xclbin_path in xclbins:
    if not os.path.exists(xclbin_path):
        continue
    
    print(f"\n📦 Testing: {os.path.basename(xclbin_path)}")
    
    try:
        xclbin = pyxrt.xclbin(xclbin_path)
        uuid = device.register_xclbin(xclbin)
        
        # Get kernels
        kernels = xclbin.get_kernels()
        print(f"   Found {len(kernels)} kernels")
        
        for k in kernels:
            name = k.get_name()
            print(f"   - {name}")
            
            # Try different instance patterns
            for suffix in ["", "_1", ":1", f":{name}_1"]:
                try:
                    kernel_name = name + suffix if suffix else name
                    kernel = pyxrt.kernel(device, uuid, kernel_name)
                    print(f"     ✅ Created kernel with: '{kernel_name}'")
                    
                    # Try to get kernel info
                    try:
                        # Test if we can allocate a simple buffer
                        bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.normal, 0)
                        print(f"        ✅ Buffer allocation works!")
                        del bo
                    except Exception as e:
                        print(f"        ❌ Buffer: {str(e)[:40]}...")
                    
                    del kernel
                    break
                except Exception as e:
                    if "not found" not in str(e):
                        print(f"     ❌ {kernel_name}: {str(e)[:30]}...")
                    
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        
print("\n📊 Summary:")
print("   The kernel instance naming is critical")
print("   Buffer allocation remains the main blocker")