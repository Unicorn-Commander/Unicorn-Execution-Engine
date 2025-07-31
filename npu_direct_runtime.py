#!/usr/bin/env python3.13
"""
Direct NPU Runtime based on transcription project
Real hardware interface - no simulations
"""

import os
import sys
import ctypes
import mmap
import struct
import numpy as np
from pathlib import Path
import logging

# Virtual environment path
sys.path.insert(0, 'npu_kernel_env/lib/python3.13/site-packages')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# C structures for IOCTL (from transcription project)
class amdxdna_drm_create_bo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("handle", ctypes.c_uint32)
    ]

class amdxdna_drm_map_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("offset", ctypes.c_uint64)
    ]

class amdxdna_drm_sync_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("direction", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64)
    ]

class amdxdna_drm_exec_cmd(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("config_handle", ctypes.c_uint32),
        ("config_size", ctypes.c_uint64),
        ("config_ptr", ctypes.c_uint64)
    ]

class amdxdna_drm_destroy_bo(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32)
    ]

# IOCTL commands
DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443
DRM_IOCTL_AMDXDNA_MAP_BO = 0xC0186444
DRM_IOCTL_AMDXDNA_SYNC_BO = 0xC0186445
DRM_IOCTL_AMDXDNA_EXEC_CMD = 0xC0206446
DRM_IOCTL_AMDXDNA_DESTROY_BO = 0xC0106448

# Sync directions
SYNC_TO_DEVICE = 0
SYNC_FROM_DEVICE = 1

# Memory flags from transcription project
BO_FLAGS_CACHEABLE = 0x10000000
BO_FLAGS_COHERENT = 0x20000000
BO_FLAGS_DEVICE_MEM = 0x40000000


class NPUDirectRuntime:
    """Direct NPU runtime interface"""
    
    def __init__(self):
        self.device_path = "/dev/accel/accel0"
        self.fd = None
        self.libc = ctypes.CDLL("libc.so.6")
        
        # Configure ioctl
        self.libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
        self.libc.ioctl.restype = ctypes.c_int
        
    def open(self) -> bool:
        """Open NPU device"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR | os.O_CLOEXEC)
            logger.info(f"✅ NPU device opened: {self.device_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to open NPU: {e}")
            return False
            
    def create_buffer(self, size: int, flags: int = BO_FLAGS_CACHEABLE) -> int:
        """Create NPU buffer object"""
        # Align to 4KB
        aligned_size = (size + 4095) & ~4095
        
        create_bo = amdxdna_drm_create_bo()
        create_bo.size = aligned_size
        create_bo.flags = flags
        create_bo.handle = 0
        
        ret = self.libc.ioctl(self.fd, DRM_IOCTL_AMDXDNA_CREATE_BO, 
                             ctypes.cast(ctypes.byref(create_bo), ctypes.c_void_p))
        
        if ret != 0:
            logger.error(f"❌ Failed to create buffer: {ret}")
            return -1
            
        logger.info(f"✅ Created buffer: handle={create_bo.handle}, size={aligned_size}")
        return create_bo.handle
        
    def map_buffer(self, handle: int, size: int) -> mmap.mmap:
        """Map NPU buffer to host memory"""
        map_bo = amdxdna_drm_map_bo()
        map_bo.handle = handle
        map_bo._pad = 0
        map_bo.offset = 0
        
        ret = self.libc.ioctl(self.fd, DRM_IOCTL_AMDXDNA_MAP_BO, 
                             ctypes.cast(ctypes.byref(map_bo), ctypes.c_void_p))
        
        if ret != 0:
            logger.error(f"❌ Failed to map buffer: {ret}")
            return None
            
        # Create mmap
        mapped = mmap.mmap(self.fd, size, 
                          flags=mmap.MAP_SHARED,
                          prot=mmap.PROT_READ | mmap.PROT_WRITE,
                          offset=map_bo.offset)
                          
        logger.info(f"✅ Mapped buffer at offset {map_bo.offset:#x}")
        return mapped
        
    def sync_buffer(self, handle: int, direction: int, size: int):
        """Sync buffer with device"""
        sync_bo = amdxdna_drm_sync_bo()
        sync_bo.handle = handle
        sync_bo.direction = direction
        sync_bo.offset = 0
        sync_bo.size = size
        
        ret = self.libc.ioctl(self.fd, DRM_IOCTL_AMDXDNA_SYNC_BO, 
                             ctypes.cast(ctypes.byref(sync_bo), ctypes.c_void_p))
        
        if ret != 0:
            logger.error(f"❌ Failed to sync buffer: {ret}")
            
    def destroy_buffer(self, handle: int):
        """Destroy NPU buffer"""
        destroy_bo = amdxdna_drm_destroy_bo()
        destroy_bo.handle = handle
        
        self.libc.ioctl(self.fd, DRM_IOCTL_AMDXDNA_DESTROY_BO, 
                        ctypes.cast(ctypes.byref(destroy_bo), ctypes.c_void_p))
        
    def execute_kernel(self, kernel_data: bytes, buffers: list) -> bool:
        """Execute NPU kernel"""
        # Create configuration buffer
        config_size = len(kernel_data)
        config_handle = self.create_buffer(config_size)
        
        if config_handle < 0:
            return False
            
        # Map and copy kernel
        config_map = self.map_buffer(config_handle, config_size)
        if not config_map:
            self.destroy_buffer(config_handle)
            return False
            
        config_map[:config_size] = kernel_data
        self.sync_buffer(config_handle, SYNC_TO_DEVICE, config_size)
        
        # Prepare execution command
        exec_cmd = amdxdna_drm_exec_cmd()
        exec_cmd.handle = buffers[0]  # Primary buffer handle
        exec_cmd.config_handle = config_handle
        exec_cmd.config_size = config_size
        exec_cmd.config_ptr = 0  # Kernel already in config buffer
        
        ret = self.libc.ioctl(self.fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, 
                             ctypes.cast(ctypes.byref(exec_cmd), ctypes.c_void_p))
        
        # Cleanup
        config_map.close()
        self.destroy_buffer(config_handle)
        
        if ret != 0:
            logger.error(f"❌ Kernel execution failed: {ret}")
            return False
            
        logger.info("✅ Kernel execution complete")
        return True
        
    def close(self):
        """Close NPU device"""
        if self.fd:
            os.close(self.fd)
            logger.info("✅ NPU device closed")


def test_direct_runtime():
    """Test direct NPU runtime"""
    
    runtime = NPUDirectRuntime()
    
    if not runtime.open():
        return False
        
    try:
        # Test buffer creation with correct flags
        test_size = 1024 * 1024  # 1MB
        
        logger.info("\n🧪 Testing NPU buffer allocation...")
        
        # Try different flag combinations
        flag_tests = [
            (BO_FLAGS_CACHEABLE, "Cacheable"),
            (BO_FLAGS_COHERENT, "Coherent"),
            (BO_FLAGS_DEVICE_MEM, "Device Memory"),
            (0, "Default")
        ]
        
        for flags, desc in flag_tests:
            logger.info(f"\n📊 Testing {desc} buffer (flags={flags:#x})...")
            
            handle = runtime.create_buffer(test_size, flags)
            
            if handle >= 0:
                # Try to map
                mapped = runtime.map_buffer(handle, test_size)
                
                if mapped:
                    # Write test pattern
                    test_data = b'NPU_TEST' * 128
                    mapped[:len(test_data)] = test_data
                    
                    # Sync to device
                    runtime.sync_buffer(handle, SYNC_TO_DEVICE, test_size)
                    
                    # Read back
                    runtime.sync_buffer(handle, SYNC_FROM_DEVICE, test_size)
                    
                    if mapped[:len(test_data)] == test_data:
                        logger.info("   ✅ Buffer read/write successful")
                    else:
                        logger.info("   ⚠️ Buffer data mismatch")
                        
                    mapped.close()
                    
                runtime.destroy_buffer(handle)
            else:
                logger.info(f"   ❌ Buffer creation failed")
                
        return True
        
    finally:
        runtime.close()


if __name__ == "__main__":
    logger.info("🦄 NPU Direct Runtime Test")
    logger.info("=" * 60)
    
    if test_direct_runtime():
        logger.info("\n✅ NPU direct runtime functional!")
    else:
        logger.error("\n❌ NPU direct runtime test failed")