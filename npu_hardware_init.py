#!/usr/bin/env python3
"""
NPU Hardware Initialization - Direct hardware access, no fallback
"""

import os
import ctypes
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class NPUHardwareInit:
    """Initialize NPU with proper library paths"""
    
    def __init__(self):
        self.npu_available = False
        self.xrt_lib = None
        self.device_handle = None
        
    def initialize(self) -> bool:
        """Initialize NPU hardware - success or fail, no fallback"""
        
        # 1. Check NPU device exists
        if not os.path.exists("/dev/accel/accel0"):
            logger.error("❌ FAIL: No NPU device at /dev/accel/accel0")
            return False
            
        logger.info("✅ NPU device found at /dev/accel/accel0")
        
        # 2. Set library paths correctly
        os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        
        # 3. Load XRT driver
        try:
            # First load dependencies
            ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_coreutil.so.2', mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2', mode=ctypes.RTLD_GLOBAL)
            
            # Then load the driver
            self.xrt_lib = ctypes.CDLL('/usr/local/xrt/lib/libxrt_driver_xdna.so')
            logger.info("✅ XRT driver loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ FAIL: Cannot load NPU driver: {e}")
            return False
            
        # 4. Initialize NPU context
        try:
            # Define XRT functions
            self.xrt_lib.xclOpen.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]
            self.xrt_lib.xclOpen.restype = ctypes.c_void_p
            
            # Open device
            device_index = 0
            log_file = b"npu.log"
            verbosity = 2
            
            self.device_handle = self.xrt_lib.xclOpen(device_index, log_file, verbosity)
            if not self.device_handle:
                raise RuntimeError("Failed to open NPU device")
                
            logger.info("✅ NPU context initialized")
            self.npu_available = True
            return True
            
        except Exception as e:
            logger.error(f"❌ FAIL: NPU initialization failed: {e}")
            return False
            
    def load_npu_kernel(self, kernel_path: str) -> bool:
        """Load compiled NPU kernel binary"""
        if not self.npu_available:
            logger.error("NPU not initialized")
            return False
            
        if not os.path.exists(kernel_path):
            logger.error(f"Kernel not found: {kernel_path}")
            return False
            
        # Load XCLBIN or compiled kernel
        # Implementation depends on kernel format
        logger.info(f"Loading NPU kernel: {kernel_path}")
        return True
        
    def cleanup(self):
        """Clean up NPU resources"""
        if self.device_handle and self.xrt_lib:
            self.xrt_lib.xclClose(self.device_handle)
            logger.info("NPU context closed")


def test_npu_init():
    """Test NPU initialization"""
    npu = NPUHardwareInit()
    
    if npu.initialize():
        logger.info("🎉 NPU READY FOR INFERENCE!")
        return True
    else:
        logger.error("💀 NPU INITIALIZATION FAILED - NO FALLBACK")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_npu_init()