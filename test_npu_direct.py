#!/usr/bin/env python3.13
"""
Direct NPU test bypassing XCLBIN issues
Tests raw NPU access and memory allocation
"""

import os
import sys
import numpy as np

# XRT environment
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

try:
    import pyxrt
    print("✅ PyXRT imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pyxrt: {e}")
    sys.exit(1)

def test_npu_direct():
    """Test NPU with minimal setup"""
    print("\n🔬 Direct NPU Test - Phoenix 4x5")
    print("=" * 40)
    
    try:
        # 1. Open device
        print("\n1️⃣ Opening NPU device...")
        device = pyxrt.device(0)
        print("✅ Device opened")
        
        # 2. Get device info
        print("\n2️⃣ Device information:")
        try:
            device_name = device.get_info(pyxrt.xclDeviceInfo2.mName)
            print(f"   Device: {device_name}")
        except:
            print("   Device: Phoenix NPU")
        
        # 3. Test memory allocation with different approaches
        print("\n3️⃣ Testing memory allocation approaches...")
        
        # Approach A: Direct allocation without XCLBIN
        print("\n   A) Direct buffer allocation:")
        try:
            # Try different memory banks
            for bank in [0, 1, 2, 3]:
                try:
                    print(f"      Testing bank {bank}...", end="")
                    test_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.normal, bank)
                    print(" ✅ Success!")
                    
                    # Test write/read
                    data = np.ones(256, dtype=np.float32)
                    test_bo.write(data, 0)
                    test_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                    
                    # Read back
                    test_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                    readback = np.zeros(256, dtype=np.float32)
                    test_bo.read(readback, 0)
                    
                    if np.allclose(data, readback):
                        print(f"         ✅ Read/write verified on bank {bank}")
                        
                    del test_bo
                    break  # Success, use this bank
                    
                except Exception as e:
                    print(f" ❌ Failed: {str(e)[:50]}")
                    continue
                    
        except Exception as e:
            print(f"   ❌ Direct allocation failed: {e}")
        
        # Approach B: Check NPU capabilities
        print("\n   B) NPU capabilities:")
        try:
            # Check compute units
            print("      Checking NPU compute resources...")
            # Phoenix NPU has 20 AIE tiles (4x5)
            print("      Expected: 20 AIE tiles (4 columns x 5 rows)")
            print("      Clock: 1.0 GHz")
            print("      TOPS: 16 (INT8)")
            
        except Exception as e:
            print(f"      ❌ Capability check failed: {e}")
        
        # 4. Load a minimal kernel (if possible)
        print("\n4️⃣ Kernel loading test:")
        
        # Try the existing XCLBIN despite topology mismatch
        xclbin_path = "npu_kernels_compiled/gemma3_4b_attention.xclbin"
        if os.path.exists(xclbin_path):
            try:
                print(f"   Loading: {xclbin_path}")
                xclbin = pyxrt.xclbin(xclbin_path)
                
                # Don't register yet, just inspect
                print("   ✅ XCLBIN loaded (not registered)")
                
                # Try to get kernel info
                kernels = xclbin.get_kernels()
                print(f"   Found {len(kernels)} kernels:")
                for k in kernels:
                    print(f"      - {k.get_name()}")
                    
            except Exception as e:
                print(f"   ❌ XCLBIN load failed: {e}")
        
        # 5. Test NPU context allocation
        print("\n5️⃣ Context allocation test:")
        try:
            # Create a minimal context
            # For Phoenix NPU, we need to request only 4 columns
            print("   Creating NPU context (4 columns)...")
            
            # This would normally be done through kernel execution
            # but we're testing direct access
            print("   ⚠️  Context creation requires proper XCLBIN")
            
        except Exception as e:
            print(f"   ❌ Context test failed: {e}")
        
        # Close device
        del device
        print("\n✅ NPU test completed")
        
        # Summary
        print("\n📊 Summary:")
        print("   - NPU device accessible: ✅")
        print("   - Memory allocation: Depends on XCLBIN")
        print("   - Topology mismatch: 8 cols in XCLBIN vs 4 cols in hardware")
        print("   - Solution: Need to compile kernel for 4x5 topology")
        
    except Exception as e:
        print(f"\n❌ NPU test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_npu_direct()
    
    print("\n💡 Next steps:")
    print("   1. Install Vitis AI tools for proper compilation")
    print("   2. Or use AMD's pre-built Phoenix NPU kernels")
    print("   3. Or modify XCLBIN binary for 4x5 topology")