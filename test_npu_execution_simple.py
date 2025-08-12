#!/usr/bin/env python3
import os
import time
import numpy as np
import pyxrt

print("🧠 Testing Real NPU Execution")
print("=============================")

os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    # Open NPU device
    device = pyxrt.device(0)
    print("✅ NPU device opened")
    
    # Load working validation kernel
    kernel_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin"
    xclbin = pyxrt.xclbin(kernel_path)
    uuid = device.register_xclbin(xclbin)
    print("✅ XCLBIN loaded")
    
    # Create kernel
    kernel = pyxrt.kernel(device, uuid, "vadd")
    print("✅ Kernel created")
    
    # Get memory banks
    banks = [kernel.group_id(i) for i in range(3)]
    print(f"💾 Memory banks: {[hex(b) for b in banks]}")
    
    # Allocate buffers
    buffer_size = 1024 * 4
    in1_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[0])
    in2_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[1])
    out_bo = pyxrt.bo(device, buffer_size, pyxrt.bo.flags.cacheable, banks[2])
    
    # Create test data
    in1_data = np.arange(1024, dtype=np.float32) * 0.1
    in2_data = np.arange(1024, dtype=np.float32) * 0.2
    expected = in1_data + in2_data
    
    # Copy to NPU
    in1_bo.write(in1_data.tobytes(), 0)
    in2_bo.write(in2_data.tobytes(), 0)
    in1_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    in2_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    print("🚀 Executing NPU kernel...")
    start_time = time.time()
    
    # Execute kernel
    run = kernel(in1_bo, in2_bo, out_bo, 1024)
    state = run.wait(10000)
    
    execution_time = (time.time() - start_time) * 1000
    print(f"✅ NPU execution completed in {execution_time:.2f} ms")
    
    # Read results
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    result_bytes = out_bo.read(buffer_size, 0)
    result = np.frombuffer(result_bytes, dtype=np.float32)
    
    # Check correctness
    max_error = np.max(np.abs(result - expected))
    print(f"📊 Max error: {max_error:.6f}")
    print(f"📊 Sample: expected {expected[0]:.3f}, got {result[0]:.3f}")
    
    if max_error < 1e-5:
        print("🎉 SUCCESS: NPU is executing correctly!")
        print(f"🚀 NPU Performance: {execution_time:.2f} ms for 1024 operations")
        
        # Calculate effective tokens/sec if this were attention
        # Rough estimate: if attention takes similar time per operation
        operations_per_token = 1024  # Simplified assumption
        tokens_per_sec = 1000 / execution_time * operations_per_token / 1000
        print(f"📊 Estimated NPU contribution: ~{tokens_per_sec:.1f} operations/ms")
        
    else:
        print("❌ NPU computation has errors")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()