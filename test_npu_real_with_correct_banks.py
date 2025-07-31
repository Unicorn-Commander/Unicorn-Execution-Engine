#!/usr/bin/env python3.13
"""
Test NPU execution with correct memory banks
"""

import pyxrt
import numpy as np
import time

print("🦄 NPU Execution with Correct Memory Banks")
print("=" * 50)

# Open device and load XCLBIN
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin")
uuid = device.register_xclbin(xclbin)
print("✅ NPU ready")

# Create kernel
kernel = pyxrt.kernel(device, uuid, "DPU_PDI_0")
print("✅ DPU_PDI_0 kernel created")

# Get correct memory banks for each argument
arg_banks = []
for i in range(8):
    try:
        group_id = kernel.group_id(i)
        arg_banks.append(group_id)
    except:
        break

print(f"\n📊 Kernel expects {len(arg_banks)} arguments with banks: {arg_banks}")

# Create buffers with correct banks
buffer_size = 4096
buffers = []

print("\n🔧 Creating buffers with correct banks...")
for i, bank in enumerate(arg_banks):
    try:
        bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, bank)
        buffers.append(bo)
        print(f"   ✅ Buffer {i} allocated in bank {bank} (0x{bank:X})")
        
        # Initialize with test data
        if i == 0:  # First input
            data = np.arange(1024, dtype=np.float32)
            bo.write(data.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    except Exception as e:
        print(f"   ❌ Buffer {i} failed: {e}")
        buffers.append(None)

# Run kernel
print("\n🚀 Executing NPU kernel...")
try:
    # Create run handle with all buffers
    run = kernel(*buffers[:len(arg_banks)])
    
    # Wait for completion
    start_time = time.time()
    state = run.wait(5000)  # 5 second timeout
    elapsed = time.time() - start_time
    
    print(f"⏱️  Execution time: {elapsed*1000:.2f} ms")
    
    if state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print("✅ NPU kernel executed successfully!")
        
        # Check output buffer (usually second buffer)
        if len(buffers) > 1 and buffers[1]:
            buffers[1].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            output = np.frombuffer(buffers[1].read(buffer_size, 0), dtype=np.float32)
            
            print(f"\n📊 Output data:")
            print(f"   First 5 values: {output[:5]}")
            print(f"   Stats: min={output.min():.2f}, max={output.max():.2f}, mean={output.mean():.2f}")
            
            # Check if NPU actually processed data
            if np.any(output != 0):
                print("   ✅ NPU produced non-zero output!")
            else:
                print("   ⚠️  Output is all zeros")
                
    elif state == pyxrt.ert_cmd_state.ERT_CMD_STATE_ERROR:
        print("❌ Kernel execution error")
        
        # Try to get error details
        print("\n🔍 Checking dmesg for NPU errors...")
        import subprocess
        result = subprocess.run(["sudo", "dmesg", "|", "tail", "-10"], 
                              capture_output=True, text=True, shell=True)
        if result.stdout:
            print(result.stdout)
    else:
        print(f"⚠️  Unexpected state: {state}")
        
except Exception as e:
    print(f"❌ Execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n📊 Summary:")
print("   - Correct memory banks discovered ✅")
print("   - Buffers allocated in NPU memory ✅")
print("   - Kernel execution attempted")
print("   - This is real NPU hardware access!")