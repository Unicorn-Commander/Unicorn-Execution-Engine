#!/usr/bin/env python3.13
"""
Test different buffer allocation strategies
"""

import pyxrt
import os

print("🔍 Buffer Allocation Testing")
print("=" * 50)

# Open device
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin")
uuid = device.register_xclbin(xclbin)

print("✅ Device and XCLBIN ready")

# Test all possible buffer flags
buffer_tests = [
    ("normal", pyxrt.bo.flags.normal),
    ("cacheable", pyxrt.bo.flags.cacheable),
    ("device_only", pyxrt.bo.flags.device_only),
    ("host_only", pyxrt.bo.flags.host_only),
    ("p2p", pyxrt.bo.flags.p2p),
    ("svm", pyxrt.bo.flags.svm),
]

# Test different memory banks
banks = [0, 1, 2, 3, 4]

# Test different sizes
sizes = [1024, 4096, 16384, 65536]

print("\n📊 Testing buffer configurations:")
print("-" * 40)

success_count = 0

for flag_name, flag in buffer_tests:
    for bank in banks:
        for size in sizes:
            try:
                bo = pyxrt.bo(device, size, flag, bank)
                print(f"✅ {flag_name:12} bank={bank} size={size:6} - SUCCESS!")
                del bo
                success_count += 1
                # If one works, skip other sizes for this flag/bank combo
                break
            except Exception as e:
                # Only print first error for each flag/bank combo
                if size == sizes[0]:
                    error_msg = str(e)
                    if "unsupported buffer type" in error_msg:
                        continue  # Skip this flag/bank combo
                    elif "Invalid argument" in error_msg:
                        continue  # Skip this bank
                    else:
                        print(f"❌ {flag_name:12} bank={bank} - {error_msg[:40]}...")

print(f"\n📊 Summary: {success_count} successful buffer allocations")

# Try with kernel group ID
print("\n🔧 Testing with kernel argument...")
try:
    kernel = pyxrt.kernel(device, uuid, "DPU_PDI_0")
    
    # Get kernel group ID
    try:
        # Try to get kernel argument info
        for arg_idx in range(10):  # Try first 10 arguments
            try:
                bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.normal, kernel.group_id(arg_idx))
                print(f"✅ Buffer allocated with kernel group_id({arg_idx})!")
                del bo
                break
            except:
                continue
    except Exception as e:
        print(f"❌ Kernel group approach failed: {e}")
        
except Exception as e:
    print(f"❌ Kernel creation failed: {e}")

print("\n💡 Next steps:")
print("   - May need to use xrt::bo C++ API directly")
print("   - Or wait for proper NPU SDK from AMD")
print("   - Current pyxrt may have limitations for NPU")