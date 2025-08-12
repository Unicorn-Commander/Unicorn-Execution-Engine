#!/usr/bin/env python3.13
"""
Test NPU with 4-column XCLBIN from AMD
"""

import os
import sys
import numpy as np

# XRT environment
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

import pyxrt

def test_4col_xclbin():
    """Test with AMD's 4-column XCLBIN"""
    print("🧪 Testing NPU with 4-column XCLBIN")
    print("=" * 50)
    
    # Try different pre-built XCLBINs
    xclbins_to_try = [
        "/opt/xilinx/xrt/amdxdna/bins/17f0_20/mobilenet_4col.xclbin",
        "/opt/xilinx/xrt/amdxdna/bins/17f0_20/preemption_4x4.xclbin",
        "/opt/xilinx/xrt/amdxdna/bins/17f0_20/gemm.xclbin",
        "/opt/xilinx/xrt/amdxdna/bins/17f0_11/mobilenet_4col.xclbin",
    ]
    
    device = pyxrt.device(0)
    print("✅ Device opened")
    
    for xclbin_path in xclbins_to_try:
        if not os.path.exists(xclbin_path):
            continue
            
        print(f"\n📦 Testing: {os.path.basename(xclbin_path)}")
        
        try:
            # Load XCLBIN
            xclbin = pyxrt.xclbin(xclbin_path)
            uuid = device.register_xclbin(xclbin)
            print("   ✅ XCLBIN registered successfully!")
            
            # Try to allocate memory
            print("   💾 Testing memory allocation...")
            try:
                # Small buffer test
                bo = pyxrt.bo(device, 4096, pyxrt.bo.flags.normal, 0)
                print("   ✅ Memory allocation successful!")
                
                # Test data transfer
                data = np.ones(1024, dtype=np.float32)
                bo.write(data, 0)
                bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                print("   ✅ Data transfer successful!")
                
                del bo
                
                # List available kernels
                kernels = xclbin.get_kernels()
                print(f"   📋 Available kernels: {len(kernels)}")
                for k in kernels:
                    print(f"      - {k.get_name()}")
                
                print(f"\n   🎯 SUCCESS! This XCLBIN works with Phoenix NPU")
                
                # Get more info
                print("\n   📊 XCLBIN Details:")
                os.system(f"/opt/xilinx/xrt/bin/xclbinutil --info --input {xclbin_path} | grep -E 'Kernels|columns|rows|tiles' | head -10")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Memory test failed: {e}")
                
        except Exception as e:
            print(f"   ❌ XCLBIN registration failed: {e}")
            continue
    
    del device
    return False

if __name__ == "__main__":
    success = test_4col_xclbin()
    
    if success:
        print("\n✅ Found working XCLBIN for Phoenix NPU!")
        print("\n💡 Next steps:")
        print("   1. Use this XCLBIN as a base for custom kernels")
        print("   2. Or adapt the kernel interface to match")
    else:
        print("\n❌ No suitable XCLBIN found")
        print("   Need to install Vitis AI for compilation")