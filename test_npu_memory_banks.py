#!/usr/bin/env python3.13
"""
Test NPU memory bank discovery
"""

import pyxrt

print("🔍 NPU Memory Bank Discovery")
print("=" * 50)

# Open device
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin")
uuid = device.register_xclbin(xclbin)

# Create kernel
kernel = pyxrt.kernel(device, uuid, "DPU_PDI_0")
print("✅ Kernel created")

# Check kernel arguments
print("\n📊 Kernel Argument Analysis:")
try:
    # Try to get argument info
    for i in range(10):
        try:
            group_id = kernel.group_id(i)
            print(f"   Arg {i}: group_id = {group_id}")
            
            # Try to allocate buffer with this group_id
            try:
                bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.cacheable, group_id)
                print(f"          ✅ Buffer allocated with group_id {group_id}")
                del bo
            except Exception as e:
                print(f"          ❌ Failed: {str(e)[:40]}...")
                
        except Exception as e:
            if i == 0:
                print(f"   Cannot get group_id: {e}")
            break
except Exception as e:
    print(f"❌ Error: {e}")

# Test special bank values
print("\n🔧 Testing special bank values:")
special_banks = [
    0x1FFFF,  # 131071 in hex
    131071,   # The bank mentioned in error
    0xFFFF,   # Common mask value
    -1,       # Sometimes means "auto"
]

for bank in special_banks:
    try:
        bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.cacheable, bank)
        print(f"✅ Bank {bank} (0x{bank:X}): SUCCESS!")
        del bo
    except Exception as e:
        print(f"❌ Bank {bank} (0x{bank:X}): {str(e)[:40]}...")

print("\n💡 The DPU kernels expect bank 131071 (0x1FFFF)")
print("   This appears to be a special NPU memory region")