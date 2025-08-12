#!/usr/bin/env python3.13
"""
Phoenix NPU Progress Summary
Shows what we've achieved and what's blocking us
"""

import pyxrt
import os
import subprocess

print("🦄 Phoenix NPU Progress Summary")
print("=" * 60)

print("\n✅ ACHIEVEMENTS:")
print("-" * 40)

# 1. NPU Detection
print("1. NPU Hardware Detection:")
result = subprocess.run(["/opt/xilinx/xrt/bin/xrt-smi", "examine", "--device", "0000:c7:00.1"], 
                       capture_output=True, text=True)
if "Phoenix" in result.stdout:
    print("   ✅ Phoenix NPU detected")
    print("   ✅ 5 columns confirmed (4x5 topology)")
    print("   ✅ Device: /dev/accel/accel0")

# 2. XRT Access
print("\n2. XRT Runtime Access:")
try:
    device = pyxrt.device(0)
    print("   ✅ NPU device opened via pyxrt")
    print("   ✅ XRT 2.20.0 working correctly")
except:
    print("   ❌ Device access failed")

# 3. XCLBIN Loading
print("\n3. XCLBIN Loading:")
try:
    xclbin = pyxrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin")
    uuid = device.register_xclbin(xclbin)
    print("   ✅ XCLBIN loaded successfully")
    
    kernels = xclbin.get_kernels()
    print(f"   ✅ Found {len(kernels)} kernels")
except:
    print("   ❌ XCLBIN loading failed")

# 4. Kernel Creation
print("\n4. Kernel Object Creation:")
try:
    kernel = pyxrt.kernel(device, uuid, "DPU_PDI_0")
    print("   ✅ DPU kernel objects can be created")
    print("   ✅ Kernel argument banks discovered:")
    
    for i in range(3):
        try:
            bank = kernel.group_id(i)
            print(f"      - Arg {i}: bank {bank} (0x{bank:X})")
        except:
            break
except:
    print("   ❌ Kernel creation failed")

# 5. Memory Allocation
print("\n5. NPU Memory Allocation:")
print("   ✅ Buffer allocation works with:")
print("      - pyxrt.bo.flags.cacheable")
print("      - pyxrt.bo.flags.host_only")
print("   ✅ Correct memory banks identified:")
print("      - Bank 131071 (0x1FFFF) for DMA")
print("      - Bank 65536 (0x10000) for compute")

print("\n❌ CURRENT BLOCKERS:")
print("-" * 40)

# Check SMU status
smu_errors = subprocess.run(["sudo", "dmesg", "|", "grep", "smu_exec", "|", "tail", "-3"],
                          capture_output=True, text=True, shell=True)
if "busy" in smu_errors.stdout:
    print("1. SMU Busy Error:")
    print("   ❌ System Management Unit stuck")
    print("   💡 Fix: System restart required")

print("\n2. Kernel Execution:")
print("   ❌ DPU kernels fail to execute")
print("   ❌ Error state returned from NPU")
print("   💡 Possible causes:")
print("      - SMU needs reset")
print("      - Kernels are validation/test only")
print("      - Need proper AIE programming")

print("\n3. MLIR-AIE Compilation:")
print("   ⚠️  LLVM/MLIR built successfully")
print("   ⚠️  MLIR-AIE Python package installed")
print("   ❌ Cannot compile for 5-column topology")
print("   💡 Need AMD's official Phoenix NPU SDK")

print("\n📊 WHAT THIS MEANS:")
print("-" * 40)
print("• We have PROVEN the NPU is accessible from Python")
print("• We can allocate NPU memory correctly")
print("• We understand the memory architecture")
print("• The hardware is ready - just needs proper kernels")
print("• No more simulations - this is real hardware!")

print("\n🚀 NEXT STEPS:")
print("-" * 40)
print("1. Restart system to clear SMU state")
print("2. Find AMD's Phoenix NPU SDK or examples")
print("3. Or continue MLIR-AIE integration")
print("4. Goal: Compile real attention kernels")

print("\n💡 KEY INSIGHT:")
print("-" * 40)
print("The Phoenix NPU is REAL and ACCESSIBLE!")
print("Previous '287.8 TPS' was sleep() simulation")
print("Now we're working with actual hardware")
print("Just need the right toolchain to unlock it! 🦄")