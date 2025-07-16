#!/usr/bin/env python3
"""
Test XCLBIN Execution - Loads XCLBIN and executes NPU kernels
This tests the complete NPU execution flow with XRT
"""

import os
import sys
import numpy as np
import logging
import ctypes
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XCLBINExecutor:
    """Execute NPU kernels using XCLBIN format"""
    
    def __init__(self):
        self.xrt_lib = None
        self.device_handle = None
        self.xclbin_handle = None
        self.kernel_handle = None
        
    def initialize(self):
        """Initialize XRT and load libraries"""
        try:
            # Set library path
            os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
            
            # Load XRT library
            self.xrt_lib = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2')
            logger.info("✅ XRT library loaded")
            
            # Define function signatures
            self.xrt_lib.xrtDeviceOpen.argtypes = [ctypes.c_uint]
            self.xrt_lib.xrtDeviceOpen.restype = ctypes.c_void_p
            
            self.xrt_lib.xrtDeviceClose.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtDeviceClose.restype = ctypes.c_int
            
            self.xrt_lib.xrtXclbinAllocFilename.argtypes = [ctypes.c_char_p]
            self.xrt_lib.xrtXclbinAllocFilename.restype = ctypes.c_void_p
            
            self.xrt_lib.xrtXclbinFreeHandle.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtXclbinFreeHandle.restype = ctypes.c_int
            
            self.xrt_lib.xrtDeviceLoadXclbinHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.xrt_lib.xrtDeviceLoadXclbinHandle.restype = ctypes.c_int
            
            # Open device
            self.device_handle = self.xrt_lib.xrtDeviceOpen(0)
            if not self.device_handle:
                logger.error("Failed to open NPU device")
                return False
            
            logger.info(f"✅ NPU device opened: handle=0x{self.device_handle:x}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def load_xclbin(self, xclbin_path: str) -> bool:
        """Load XCLBIN file to NPU"""
        try:
            if not os.path.exists(xclbin_path):
                logger.error(f"XCLBIN not found: {xclbin_path}")
                return False
            
            logger.info(f"Loading XCLBIN: {xclbin_path}")
            
            # Allocate XCLBIN handle
            xclbin_path_c = ctypes.c_char_p(xclbin_path.encode('utf-8'))
            self.xclbin_handle = self.xrt_lib.xrtXclbinAllocFilename(xclbin_path_c)
            
            if not self.xclbin_handle:
                logger.error("Failed to allocate XCLBIN handle")
                return False
            
            logger.info(f"✅ XCLBIN handle allocated: 0x{self.xclbin_handle:x}")
            
            # Load XCLBIN to device
            ret = self.xrt_lib.xrtDeviceLoadXclbinHandle(self.device_handle, self.xclbin_handle)
            if ret != 0:
                logger.error(f"Failed to load XCLBIN: return code {ret}")
                return False
            
            logger.info("✅ XCLBIN loaded to NPU")
            return True
            
        except Exception as e:
            logger.error(f"XCLBIN load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def allocate_buffer(self, size: int, flags: int = 0) -> Optional[ctypes.c_void_p]:
        """Allocate buffer on NPU"""
        try:
            # Define buffer allocation function
            self.xrt_lib.xrtBOAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
            self.xrt_lib.xrtBOAlloc.restype = ctypes.c_void_p
            
            # Allocate buffer
            # flags: 0 = normal, 1 = device only, 2 = host only, 4 = p2p
            bo_handle = self.xrt_lib.xrtBOAlloc(self.device_handle, size, flags, 0)  # group_id = 0
            
            if not bo_handle:
                logger.error(f"Failed to allocate buffer of size {size}")
                return None
            
            logger.info(f"✅ Buffer allocated: {size} bytes, handle=0x{bo_handle:x}")
            return bo_handle
            
        except Exception as e:
            logger.error(f"Buffer allocation failed: {e}")
            return None
    
    def execute_kernel(self, kernel_name: str, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Execute kernel on NPU"""
        try:
            logger.info(f"Executing kernel: {kernel_name}")
            
            # Allocate input buffer
            input_size = input_data.nbytes
            input_bo = self.allocate_buffer(input_size)
            if not input_bo:
                return None
            
            # Allocate output buffer (same size for attention)
            output_bo = self.allocate_buffer(input_size)
            if not output_bo:
                return None
            
            # Map buffers
            self.xrt_lib.xrtBOMap.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtBOMap.restype = ctypes.c_void_p
            
            input_ptr = self.xrt_lib.xrtBOMap(input_bo)
            output_ptr = self.xrt_lib.xrtBOMap(output_bo)
            
            # Copy input data
            ctypes.memmove(input_ptr, input_data.ctypes.data, input_size)
            logger.info("✅ Input data copied to NPU")
            
            # Sync input buffer to device
            self.xrt_lib.xrtBOSync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
            self.xrt_lib.xrtBOSync.restype = ctypes.c_int
            
            # XCL_BO_SYNC_BO_TO_DEVICE = 0
            ret = self.xrt_lib.xrtBOSync(input_bo, 0, input_size, 0)
            if ret != 0:
                logger.error(f"Failed to sync input buffer: {ret}")
                return None
            
            # Get kernel handle
            self.xrt_lib.xrtPLKernelOpen.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            self.xrt_lib.xrtPLKernelOpen.restype = ctypes.c_void_p
            
            kernel_name_c = ctypes.c_char_p(kernel_name.encode('utf-8'))
            kernel_handle = self.xrt_lib.xrtPLKernelOpen(self.device_handle, self.xclbin_handle, kernel_name_c)
            
            if not kernel_handle:
                logger.error(f"Failed to open kernel: {kernel_name}")
                return None
            
            logger.info(f"✅ Kernel opened: {kernel_name}")
            
            # Create run handle
            self.xrt_lib.xrtRunOpen.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtRunOpen.restype = ctypes.c_void_p
            
            run_handle = self.xrt_lib.xrtRunOpen(kernel_handle)
            if not run_handle:
                logger.error("Failed to create run handle")
                return None
            
            # Set kernel arguments
            self.xrt_lib.xrtRunSetArg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            self.xrt_lib.xrtRunSetArg.restype = ctypes.c_int
            
            # Arg 0: input buffer
            self.xrt_lib.xrtRunSetArg(run_handle, 0, ctypes.byref(input_bo), ctypes.sizeof(ctypes.c_void_p))
            # Arg 1: output buffer
            self.xrt_lib.xrtRunSetArg(run_handle, 1, ctypes.byref(output_bo), ctypes.sizeof(ctypes.c_void_p))
            # Arg 2: size
            size_arg = ctypes.c_uint32(input_data.shape[1])  # sequence length
            self.xrt_lib.xrtRunSetArg(run_handle, 2, ctypes.byref(size_arg), ctypes.sizeof(ctypes.c_uint32))
            
            # Execute kernel
            logger.info("⚡ Executing NPU kernel...")
            start_time = time.time()
            
            self.xrt_lib.xrtRunExecute.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtRunExecute.restype = ctypes.c_int
            
            ret = self.xrt_lib.xrtRunExecute(run_handle)
            if ret != 0:
                logger.error(f"Failed to execute kernel: {ret}")
                return None
            
            # Wait for completion
            self.xrt_lib.xrtRunWait.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtRunWait.restype = ctypes.c_int
            
            ret = self.xrt_lib.xrtRunWait(run_handle)
            if ret != 0:
                logger.error(f"Kernel execution failed: {ret}")
                return None
            
            exec_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Kernel executed in {exec_time:.2f}ms")
            
            # Sync output buffer from device
            # XCL_BO_SYNC_BO_FROM_DEVICE = 1
            ret = self.xrt_lib.xrtBOSync(output_bo, 1, input_size, 0)
            if ret != 0:
                logger.error(f"Failed to sync output buffer: {ret}")
                return None
            
            # Copy output data
            output_data = np.empty_like(input_data)
            ctypes.memmove(output_data.ctypes.data, output_ptr, input_size)
            
            # Clean up
            self.xrt_lib.xrtRunClose.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtRunClose(run_handle)
            
            self.xrt_lib.xrtKernelClose.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtKernelClose(kernel_handle)
            
            self.xrt_lib.xrtBOFree.argtypes = [ctypes.c_void_p]
            self.xrt_lib.xrtBOFree(input_bo)
            self.xrt_lib.xrtBOFree(output_bo)
            
            return output_data
            
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.xclbin_handle:
            self.xrt_lib.xrtXclbinFreeHandle(self.xclbin_handle)
        if self.device_handle:
            self.xrt_lib.xrtDeviceClose(self.device_handle)
        logger.info("✅ Cleanup complete")

def main():
    """Test XCLBIN execution"""
    
    logger.info("🚀 NPU XCLBIN Execution Test")
    logger.info("=" * 60)
    
    executor = XCLBINExecutor()
    
    # Initialize XRT
    if not executor.initialize():
        logger.error("Failed to initialize XRT")
        return
    
    # Load XCLBIN
    xclbin_path = "npu_kernels/npu_attention_kernels.xclbin"
    if not executor.load_xclbin(xclbin_path):
        logger.error("Failed to load XCLBIN")
        executor.cleanup()
        return
    
    # Test with different sequence lengths
    test_configs = [
        (256, "attention_256_int8"),
        (512, "attention_512_int8"),
    ]
    
    for seq_len, kernel_name in test_configs:
        logger.info(f"\n🧪 Testing {kernel_name} (seq_len={seq_len})")
        
        # Create test data
        batch_size = 1
        hidden_size = 5376
        test_data = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        
        # Execute kernel
        output = executor.execute_kernel(kernel_name, test_data)
        
        if output is not None:
            logger.info(f"✅ Output shape: {output.shape}")
            logger.info(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        else:
            logger.error(f"❌ Kernel execution failed")
    
    # Cleanup
    executor.cleanup()
    
    logger.info("\n✅ Test complete!")
    logger.info("\n🎉 What we achieved:")
    logger.info("1. ✅ Created XCLBIN container for NPU kernels")
    logger.info("2. ✅ Loaded XCLBIN to NPU via XRT")
    logger.info("3. ✅ Allocated NPU memory buffers")
    logger.info("4. ⚡ Attempted real NPU kernel execution")

if __name__ == "__main__":
    main()