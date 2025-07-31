
import os
import pyxrt
import numpy as np

# Open device
device = pyxrt.device(0)
print("✅ Device opened")

# Load XCLBIN
xclbin = pyxrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm.xclbin")
uuid = device.register_xclbin(xclbin)
print("✅ XCLBIN registered")

# List kernels
kernels = xclbin.get_kernels()
print(f"\n📋 Available kernels: {len(kernels)}")
for k in kernels:
    print(f"   - {k.get_name()}")

# Try to allocate memory
try:
    bo = pyxrt.bo(device, 4096, pyxrt.bo.flags.normal, 0)
    print("\n✅ Memory allocation successful!")
    
    # Write test data
    data = np.ones(1024, dtype=np.float32)
    bo.write(data, 0)
    bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    print("✅ Data transfer successful!")
    
except Exception as e:
    print(f"❌ Memory test failed: {e}")
