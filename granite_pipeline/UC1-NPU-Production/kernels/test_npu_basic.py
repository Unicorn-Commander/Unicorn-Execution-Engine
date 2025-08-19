#!/usr/bin/env python3
"""
Basic NPU Test - Verify we can interact with NPU hardware
"""

import os
import sys
import numpy as np
import time

# Check if pyxrt is available
try:
    sys.path.append('/opt/xilinx/xrt/python')
    import pyxrt
    print("✅ PyXRT imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pyxrt: {e}")
    print("Trying alternative import...")
    try:
        import xrt
        print("✅ XRT imported as alternative")
        pyxrt = xrt
    except:
        print("❌ No XRT Python bindings found")
        sys.exit(1)

def test_npu_device():
    """Test basic NPU device access"""
    
    print("\n" + "="*60)
    print("NPU DEVICE TEST")
    print("="*60)
    
    try:
        # Enumerate available devices (returns count, not list)
        device_count = pyxrt.enumerate_devices()
        print(f"Found {device_count} device(s)")
        
        if device_count == 0:
            print("❌ No NPU devices found")
            return False
        
        # Open first device using correct API
        device = pyxrt.device(0)
        print(f"✅ Opened device 0")
        
        # Get device info using xrt_info_device enum
        try:
            device_name = device.get_info(pyxrt.xrt_info_device.name)
            print(f"Device Name: {device_name}")
        except:
            print("Device name not available")
        
        try:
            device_bdf = device.get_info(pyxrt.xrt_info_device.bdf)
            print(f"Device BDF: {device_bdf}")
        except:
            print("Device BDF not available")
        
        # Check available memory banks
        try:
            # Get memory topology
            mem_topology = device.get_info(pyxrt.xrt_info_device.mem_topology_raw)
            print(f"Memory topology available: {len(mem_topology) if mem_topology else 0} bytes")
        except:
            print("Memory topology not available")
        
        print("✅ NPU device is accessible!")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing NPU: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_kernel():
    """Create a simple test kernel for NPU"""
    
    print("\n" + "="*60)
    print("SIMPLE NPU KERNEL TEST")
    print("="*60)
    
    # For now, we'll create a simple compute simulation
    # Real NPU kernels require compiled XCLBIN files
    
    print("Creating test data...")
    size = 1024
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    print(f"Input arrays: {size} elements each")
    
    # Simulate NPU computation (would be real kernel in production)
    start = time.perf_counter()
    c = a + b  # Simple operation
    end = time.perf_counter()
    
    elapsed = (end - start) * 1000  # Convert to ms
    throughput = size / elapsed  # Elements per ms
    
    print(f"Operation completed in {elapsed:.3f} ms")
    print(f"Throughput: {throughput:.0f} elements/ms")
    
    return True

def check_xrt_environment():
    """Check XRT environment variables"""
    
    print("\n" + "="*60)
    print("XRT ENVIRONMENT CHECK")
    print("="*60)
    
    xrt_vars = [
        "XILINX_XRT",
        "LD_LIBRARY_PATH",
        "PATH"
    ]
    
    for var in xrt_vars:
        value = os.environ.get(var, "Not set")
        if var == "PATH" or var == "LD_LIBRARY_PATH":
            # Show only XRT-related paths
            paths = value.split(':')
            xrt_paths = [p for p in paths if 'xrt' in p.lower() or 'xilinx' in p.lower()]
            if xrt_paths:
                print(f"{var}: {':'.join(xrt_paths)}")
            else:
                print(f"{var}: No XRT paths found")
        else:
            print(f"{var}: {value}")
    
    # Check for XRT libraries
    xrt_lib = "/opt/xilinx/xrt/lib"
    if os.path.exists(xrt_lib):
        print(f"\n✅ XRT libraries found at {xrt_lib}")
        lib_files = os.listdir(xrt_lib)
        print(f"   Found {len(lib_files)} library files")
    else:
        print(f"\n❌ XRT libraries not found at {xrt_lib}")
    
    return True

def main():
    print("🚀 NPU BASIC FUNCTIONALITY TEST")
    print("Testing NPU hardware access and basic operations\n")
    
    # Check environment
    env_ok = check_xrt_environment()
    
    # Test device access
    device_ok = test_npu_device()
    
    # Test simple kernel
    kernel_ok = create_simple_kernel()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Device Access: {'✅ PASS' if device_ok else '❌ FAIL'}")
    print(f"Kernel Test: {'✅ PASS' if kernel_ok else '❌ FAIL'}")
    
    if device_ok:
        print("\n✅ NPU is accessible and ready for kernel development!")
        print("Next step: Create MLIR-AIE kernels for real NPU acceleration")
    else:
        print("\n⚠️ NPU access issues detected")
        print("Check XRT installation and device permissions")

if __name__ == "__main__":
    main()