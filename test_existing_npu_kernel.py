#!/usr/bin/env python3.13
"""
Test the existing NPU attention kernel
"""

import os
import sys
import numpy as np

os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

import pyxrt

def test_existing_kernel():
    """Test the pre-compiled NPU attention kernel"""
    print("🧪 Testing Existing NPU Attention Kernel")
    print("=" * 50)
    
    device = pyxrt.device(0)
    print("✅ Device opened")
    
    # Test the existing compiled kernel
    xclbin_path = "/home/ucadmin/Development/npu_kernels/npu_attention_kernels.xclbin"
    
    if not os.path.exists(xclbin_path):
        print(f"❌ XCLBIN not found: {xclbin_path}")
        return False
    
    print(f"\n📦 Loading: {xclbin_path}")
    print(f"   Size: {os.path.getsize(xclbin_path) / 1024:.1f} KB")
    
    try:
        # Load and register XCLBIN
        xclbin = pyxrt.xclbin(xclbin_path)
        uuid = device.register_xclbin(xclbin)
        print("✅ XCLBIN registered successfully!")
        
        # Get kernel info
        kernels = xclbin.get_kernels()
        print(f"\n📋 Found {len(kernels)} kernels:")
        
        kernel_names = []
        for k in kernels:
            kname = k.get_name()
            kernel_names.append(kname)
            print(f"   - {kname}")
            
            # Get kernel args
            try:
                args = []
                for i in range(10):  # Try up to 10 args
                    try:
                        arg = k.get_arg(i)
                        args.append(arg.get_name())
                    except:
                        break
                if args:
                    print(f"     Args: {', '.join(args)}")
            except:
                pass
        
        # Try to create kernel objects
        print("\n🔧 Creating kernel objects...")
        
        for kname in kernel_names:
            try:
                print(f"   {kname}: ", end="")
                kernel = pyxrt.kernel(device, uuid, kname)
                print("✅ Success")
                
                # Try to run a simple test
                if "attention" in kname.lower():
                    print("      Testing attention kernel...")
                    
                    # Allocate test buffers
                    size = 256 * 4  # 256 float32 values
                    
                    try:
                        # Input buffer
                        in_bo = pyxrt.bo(device, size, pyxrt.bo.flags.normal, 0)
                        print("      ✅ Input buffer allocated")
                        
                        # Output buffer  
                        out_bo = pyxrt.bo(device, size, pyxrt.bo.flags.normal, 0)
                        print("      ✅ Output buffer allocated")
                        
                        # Fill input with test data
                        in_data = np.random.randn(256).astype(np.float32)
                        in_bo.write(in_data, 0)
                        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                        
                        # Run kernel
                        run = kernel(in_bo, out_bo, 256)
                        run.wait()
                        
                        # Get output
                        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                        out_data = np.zeros(256, dtype=np.float32)
                        out_bo.read(out_data, 0)
                        
                        print("      ✅ Kernel execution successful!")
                        print(f"      Input sum: {in_data.sum():.3f}")
                        print(f"      Output sum: {out_data.sum():.3f}")
                        
                    except Exception as e:
                        print(f"      ❌ Execution failed: {e}")
                        
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ XCLBIN registration failed: {e}")
        
        # Check if it's the column mismatch issue
        if "exceed" in str(e) or "columns" in str(e):
            print("\n⚠️  Topology mismatch detected")
            print("   XCLBIN expects different NPU configuration")
            print("   This kernel may be compiled for a different NPU model")
    
    return False

if __name__ == "__main__":
    success = test_existing_kernel()
    
    if success:
        print("\n✅ NPU kernel test successful!")
    else:
        print("\n💡 Solutions:")
        print("   1. The existing kernel may be for a different NPU topology")
        print("   2. We can test iGPU acceleration independently")
        print("   3. Or use CPU-only mode with optimizations")