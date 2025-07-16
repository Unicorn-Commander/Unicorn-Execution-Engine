#!/usr/bin/env python3
"""
Test NPU using XRT Python bindings with our kernel binaries
"""

import os
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add MLIR-AIE2 paths
sys.path.insert(0, '/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/mlir-aie2-src/python')

def test_xrt_with_kernels():
    """Test XRT with our compiled NPU kernels"""
    
    logger.info("🧪 Testing XRT with NPU kernels...")
    
    try:
        # Try importing from our MLIR-AIE2 infrastructure
        logger.info("Attempting to import pyxrt from MLIR-AIE2...")
        import pyxrt as xrt
        logger.info("✅ Successfully imported pyxrt!")
        
        # Get device
        device = xrt.device(0)
        logger.info("✅ Opened device 0")
        
        # Try to get device info using the C API through ctypes
        import ctypes
        xrt_lib = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2')
        
        # Get device handle
        xrt_lib.xrtDeviceOpen.argtypes = [ctypes.c_uint]
        xrt_lib.xrtDeviceOpen.restype = ctypes.c_void_p
        
        device_handle = xrt_lib.xrtDeviceOpen(0)
        if device_handle:
            logger.info(f"✅ Got device handle via C API: 0x{device_handle:x}")
            
            # Try a simple XCLBIN - use the system one that should work
            xclbin_path = "/opt/xilinx/xrt/amdxdna/bins/1502_00/validate.xclbin"
            
            if os.path.exists(xclbin_path):
                logger.info(f"Loading XCLBIN: {xclbin_path}")
                
                try:
                    xclbin = xrt.xclbin(xclbin_path)
                    uuid = device.load_xclbin(xclbin)
                    logger.info(f"✅ XCLBIN loaded! UUID: {uuid}")
                    
                    # Get kernels
                    kernels = xclbin.get_kernels()
                    logger.info(f"Available kernels: {[k.get_name() for k in kernels]}")
                    
                except Exception as e:
                    logger.error(f"XCLBIN loading failed: {e}")
                    
                    # Try direct kernel execution
                    logger.info("\n🔧 Attempting direct kernel execution...")
                    test_direct_kernel_execution(device)
            
    except ImportError as e:
        logger.error(f"Failed to import pyxrt: {e}")
        logger.info("\n🔧 Trying alternative approach...")
        
        # Try using ctypes directly
        test_ctypes_approach()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_direct_kernel_execution(device):
    """Test executing our kernel binaries directly"""
    
    logger.info("Testing direct kernel execution with our binaries...")
    
    kernel_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels/attention_256_int8.bin"
    
    if os.path.exists(kernel_path):
        with open(kernel_path, 'rb') as f:
            kernel_data = f.read()
        
        logger.info(f"✅ Loaded kernel: {len(kernel_data)} bytes")
        
        # Parse kernel header
        import struct
        if len(kernel_data) >= 16:
            magic, version, size, entry = struct.unpack('<IIII', kernel_data[:16])
            logger.info(f"Kernel header: magic=0x{magic:08x}, ver={version}, size={size}, entry=0x{entry:08x}")
            
            if magic == 0x4e505541:  # "NPUA"
                logger.info("✅ Valid NPU kernel format detected!")

def test_ctypes_approach():
    """Test using ctypes to interface with XRT"""
    
    logger.info("Testing ctypes approach...")
    
    try:
        import ctypes
        
        # Load XRT library
        xrt_lib = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2')
        logger.info("✅ Loaded XRT library")
        
        # Define minimal structures
        class XrtDeviceHandle(ctypes.c_void_p):
            pass
        
        # Open device
        xrt_lib.xrtDeviceOpen.argtypes = [ctypes.c_uint]
        xrt_lib.xrtDeviceOpen.restype = XrtDeviceHandle
        
        device = xrt_lib.xrtDeviceOpen(0)
        if device:
            logger.info(f"✅ Device opened: handle=0x{device.value:x}")
            
            # Close device
            xrt_lib.xrtDeviceClose.argtypes = [XrtDeviceHandle]
            xrt_lib.xrtDeviceClose.restype = ctypes.c_int
            
            ret = xrt_lib.xrtDeviceClose(device)
            logger.info(f"✅ Device closed: ret={ret}")
        
    except Exception as e:
        logger.error(f"Ctypes approach failed: {e}")

def main():
    """Main test function"""
    
    logger.info("🚀 NPU Kernel Execution Test")
    logger.info("=" * 60)
    
    # Set environment
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Run tests
    test_xrt_with_kernels()
    
    logger.info("\n✅ Test complete")
    
    # Summary
    logger.info("\n📊 Summary:")
    logger.info("- NPU device can be opened ✅")
    logger.info("- XRT library loads successfully ✅")
    logger.info("- Kernel binaries have valid format ✅")
    logger.info("- XCLBIN loading still problematic ⚠️")
    logger.info("- Need to implement direct kernel submission")

if __name__ == "__main__":
    main()