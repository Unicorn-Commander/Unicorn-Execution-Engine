#!/usr/bin/env python3.13
"""
Test different buffer allocation flags for NPU
"""

import os
import sys
import numpy as np

os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

import pyxrt

def test_buffer_flags():
    """Test all possible buffer allocation approaches"""
    print("🧪 Testing NPU Buffer Allocation Flags")
    print("=" * 50)
    
    device = pyxrt.device(0)
    
    # First register a simple XCLBIN
    xclbin_path = "/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin"
    xclbin = pyxrt.xclbin(xclbin_path)
    uuid = device.register_xclbin(xclbin)
    print(f"✅ Registered XCLBIN: {os.path.basename(xclbin_path)}")
    
    # Get memory topology info
    print("\n📊 Memory topology:")
    mem_topology = xclbin.get_mem_topology()
    print(f"   Memory regions: {len(mem_topology)}")
    
    for idx, mem in enumerate(mem_topology):
        print(f"   [{idx}] Tag: {mem.m_tag}, Size: {mem.m_size}, Used: {mem.m_used}")
    
    # Test different buffer allocation approaches
    print("\n🔬 Testing buffer allocations:")
    
    # All possible flag combinations
    flag_tests = [
        ("normal", pyxrt.bo.flags.normal),
        ("cacheable", pyxrt.bo.flags.cacheable),
        ("device_only", pyxrt.bo.flags.device_only),
        ("host_only", pyxrt.bo.flags.host_only),
        ("p2p", pyxrt.bo.flags.p2p),
        ("svm", pyxrt.bo.flags.svm),
    ]
    
    # Test each flag with different memory banks
    for flag_name, flag in flag_tests:
        print(f"\n   Testing {flag_name} flag:")
        
        for bank in range(len(mem_topology)):
            if mem_topology[bank].m_used == 0:
                continue
                
            try:
                print(f"      Bank {bank}: ", end="")
                
                # Try small allocation
                bo = pyxrt.bo(device, 1024, flag, bank)
                print("✅ Success!")
                
                # Test write/read if possible
                if flag_name not in ["device_only"]:
                    try:
                        data = np.ones(256, dtype=np.uint8)
                        bo.write(data, 0)
                        print(f"         Write: ✅")
                    except:
                        print(f"         Write: ❌")
                
                del bo
                
            except Exception as e:
                print(f"❌ {str(e)[:40]}...")
    
    # Try getting kernel handles
    print("\n📋 Available kernels:")
    kernels = xclbin.get_kernels()
    for k in kernels:
        kname = k.get_name()
        print(f"   - {kname}")
        
        # Try to create kernel object
        try:
            kernel = pyxrt.kernel(device, uuid, kname)
            print(f"     ✅ Kernel object created")
            
            # Get argument info
            try:
                num_args = k.get_num_args()
                print(f"     Arguments: {num_args}")
            except:
                pass
                
        except Exception as e:
            print(f"     ❌ Failed to create kernel: {e}")
    
    del device
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_buffer_flags()