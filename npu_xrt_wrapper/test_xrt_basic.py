#!/usr/bin/env python3
"""
Basic XRT test to verify NPU can be accessed
"""

import ctypes
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xrt_basic():
    """Test basic XRT functionality"""
    
    logger.info("🧪 Testing Basic XRT Functionality...")
    
    try:
        # Load XRT library
        os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        xrt_core = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2', mode=ctypes.RTLD_GLOBAL)
        
        logger.info("✅ XRT library loaded")
        
        # Test device open
        xrt_core.xrtDeviceOpen.argtypes = [ctypes.c_uint]
        xrt_core.xrtDeviceOpen.restype = ctypes.c_void_p
        
        device_handle = xrt_core.xrtDeviceOpen(0)
        
        if device_handle:
            logger.info("✅ NPU device opened successfully")
            
            # Get device info
            xrt_core.xrtDeviceGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            xrt_core.xrtDeviceGetInfo.restype = ctypes.c_size_t
            
            # Try to get device name (info type 0)
            info_buffer = ctypes.create_string_buffer(256)
            xrt_core.xrtDeviceGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t]
            xrt_core.xrtDeviceGetInfo.restype = ctypes.c_int
            
            result = xrt_core.xrtDeviceGetInfo(device_handle, 0, info_buffer, 256)
            if result == 0:
                logger.info(f"Device info: {info_buffer.value}")
            
            # Test buffer allocation with different flags
            xrt_core.xrtBOAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint]
            xrt_core.xrtBOAlloc.restype = ctypes.c_void_p
            
            # Try different flags
            test_flags = [
                (0x00000000, "NONE"),
                (0x00000001, "CACHEABLE"),
                (0x00000010, "NORMAL"),
                (0x00010000, "DEVICE_ONLY"),
                (0x00020000, "HOST_ONLY"),
            ]
            
            for flag, name in test_flags:
                buffer_handle = xrt_core.xrtBOAlloc(device_handle, 1024, flag, 0)
                if buffer_handle:
                    logger.info(f"✅ Buffer allocated with flag {name} (0x{flag:08x})")
                    # Free the buffer
                    xrt_core.xrtBOFree.argtypes = [ctypes.c_void_p]
                    xrt_core.xrtBOFree.restype = ctypes.c_int
                    xrt_core.xrtBOFree(buffer_handle)
                else:
                    logger.warning(f"❌ Failed to allocate buffer with flag {name} (0x{flag:08x})")
            
            # Close device
            xrt_core.xrtDeviceClose.argtypes = [ctypes.c_void_p]
            xrt_core.xrtDeviceClose.restype = ctypes.c_int
            xrt_core.xrtDeviceClose(device_handle)
            
            logger.info("✅ Device closed successfully")
            
        else:
            logger.error("❌ Failed to open NPU device")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_xrt_basic()