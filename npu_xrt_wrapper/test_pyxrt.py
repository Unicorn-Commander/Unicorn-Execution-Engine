#!/usr/bin/env python3
"""
Test NPU using official PyXRT Python bindings
"""

import sys
import os
import logging
import numpy as np

# Add XRT Python path
sys.path.insert(0, '/opt/xilinx/xrt/python')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pyxrt():
    """Test NPU using PyXRT"""
    
    logger.info("🧪 Testing NPU with PyXRT...")
    
    try:
        import pyxrt
        logger.info("✅ PyXRT module loaded")
        
        # Get device
        device = pyxrt.device(0)
        logger.info(f"✅ Opened device 0")
        
        # Get device info
        device_name = device.get_info(pyxrt.info.device.NAME)
        logger.info(f"Device name: {device_name}")
        
        # Test XCLBIN loading with validate.xclbin
        xclbin_path = "/opt/xilinx/xrt/amdxdna/bins/1502_00/validate.xclbin"
        
        if os.path.exists(xclbin_path):
            logger.info(f"Loading XCLBIN: {xclbin_path}")
            
            # Load XCLBIN
            xclbin = pyxrt.xclbin(xclbin_path)
            uuid = device.load_xclbin(xclbin)
            logger.info(f"✅ XCLBIN loaded, UUID: {uuid}")
            
            # Get kernels
            kernels = xclbin.get_kernels()
            logger.info(f"Available kernels: {[k.get_name() for k in kernels]}")
            
            # Try to create kernel
            if kernels:
                kernel_name = kernels[0].get_name()
                logger.info(f"Creating kernel: {kernel_name}")
                
                kernel = pyxrt.kernel(device, uuid, kernel_name)
                logger.info("✅ Kernel created")
                
                # Test buffer allocation
                test_size = 1024
                bo_in = pyxrt.bo(device, test_size, pyxrt.bo.normal, kernel.group_id(0))
                bo_out = pyxrt.bo(device, test_size, pyxrt.bo.normal, kernel.group_id(1))
                
                logger.info("✅ Buffers allocated")
                
                # Write test data
                test_data = np.random.rand(test_size // 4).astype(np.float32)
                bo_in.write(test_data.tobytes())
                bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                
                logger.info("✅ Data written to device")
                
                # Create run handle
                run = kernel(bo_in, bo_out, test_size)
                run.wait()
                
                logger.info("✅ Kernel executed")
                
                # Read results
                bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                output = np.frombuffer(bo_out.read(test_size), dtype=np.float32)
                
                logger.info(f"✅ Results read: shape={output.shape}")
                logger.info(f"   Mean: {output.mean():.4f}, Std: {output.std():.4f}")
                
        else:
            logger.error(f"XCLBIN not found: {xclbin_path}")
            
    except ImportError as e:
        logger.error(f"Failed to import pyxrt: {e}")
        logger.info("Trying alternative import...")
        
        # Try importing as a shared library
        import ctypes
        pyxrt_so = ctypes.CDLL('/opt/xilinx/xrt/python/pyxrt.cpython-313-x86_64-linux-gnu.so')
        logger.info("Loaded pyxrt.so directly, but can't use Python API this way")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pyxrt()