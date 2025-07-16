#!/usr/bin/env python3
"""
NPU Kernel Executor - Direct kernel execution using ctypes
Since XRT headers are not available, we'll interface directly with the XRT library
"""

import ctypes
import numpy as np
import os
import logging
from typing import Optional, Tuple, List
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUKernelExecutor:
    """Execute NPU kernels using direct XRT library calls"""
    
    def __init__(self):
        self.xrt_lib = None
        self.device_handle = None
        self.kernel_handle = None
        self.buffers = {}
        
        # Load XRT libraries
        self._load_xrt_libraries()
        
    def _load_xrt_libraries(self):
        """Load XRT shared libraries"""
        try:
            # Set library path
            os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
            
            # Load core XRT library
            self.xrt_core = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2', mode=ctypes.RTLD_GLOBAL)
            self.xrt_coreutil = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_coreutil.so.2', mode=ctypes.RTLD_GLOBAL)
            
            logger.info("✅ XRT libraries loaded successfully")
            
            # Define function signatures
            self._setup_function_signatures()
            
        except Exception as e:
            logger.error(f"Failed to load XRT libraries: {e}")
            raise
            
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for XRT API"""
        
        # Define buffer flags based on XRT constants
        # These are the actual XRT buffer flags from xrt.h
        self.XCL_BO_FLAGS_NONE = 0
        self.XCL_BO_FLAGS_CACHEABLE = 1 << 0  # 0x1
        self.XCL_BO_FLAGS_DEVICE_RAM = 1 << 1  # 0x2
        self.XCL_BO_FLAGS_HOST_ONLY = 1 << 2   # 0x4
        self.XCL_BO_FLAGS_P2P = 1 << 3         # 0x8
        self.XCL_BO_FLAGS_SVM = 1 << 4         # 0x10
        
        # Use DEVICE_RAM flag for NPU buffers
        self.DEFAULT_BO_FLAGS = self.XCL_BO_FLAGS_DEVICE_RAM
        
        # Sync directions
        self.XCL_BO_SYNC_BO_TO_DEVICE = 0
        self.XCL_BO_SYNC_BO_FROM_DEVICE = 1
        
        # Device management
        # xrtDeviceOpen(unsigned int index)
        self.xrt_core.xrtDeviceOpen.argtypes = [ctypes.c_uint]
        self.xrt_core.xrtDeviceOpen.restype = ctypes.c_void_p
        
        # xrtDeviceClose(xrtDeviceHandle dhdl)
        self.xrt_core.xrtDeviceClose.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtDeviceClose.restype = ctypes.c_int
        
        # xrtDeviceGetXclbinUUID(xrtDeviceHandle dhdl, xuid_t out)
        self.xrt_core.xrtDeviceGetXclbinUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.xrt_core.xrtDeviceGetXclbinUUID.restype = None
        
        # XCLBIN loading
        # xrtDeviceLoadXclbin(xrtDeviceHandle dhdl, const void* buffer)
        self.xrt_core.xrtDeviceLoadXclbin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.xrt_core.xrtDeviceLoadXclbin.restype = ctypes.c_int
        
        # Buffer management
        # xrtBOAlloc(xrtDeviceHandle dhdl, size_t size, xrtBufferFlags flags, unsigned int bank)
        self.xrt_core.xrtBOAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint]
        self.xrt_core.xrtBOAlloc.restype = ctypes.c_void_p
        
        # xrtBOFree(xrtBufferHandle bhdl)
        self.xrt_core.xrtBOFree.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtBOFree.restype = ctypes.c_int
        
        # xrtBOWrite(xrtBufferHandle bhdl, const void* src, size_t size, size_t seek)
        self.xrt_core.xrtBOWrite.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.xrt_core.xrtBOWrite.restype = ctypes.c_int
        
        # xrtBORead(xrtBufferHandle bhdl, void* dst, size_t size, size_t skip)
        self.xrt_core.xrtBORead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.xrt_core.xrtBORead.restype = ctypes.c_int
        
        # xrtBOSync(xrtBufferHandle bhdl, xrtBOSyncDirection dir, size_t size, size_t offset)
        self.xrt_core.xrtBOSync.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t]
        self.xrt_core.xrtBOSync.restype = ctypes.c_int
        
        # xrtBOMap(xrtBufferHandle bhdl)
        self.xrt_core.xrtBOMap.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtBOMap.restype = ctypes.c_void_p
        
        # Kernel management  
        # xrtPLKernelOpen(xrtDeviceHandle dhdl, const xuid_t xclbinId, const char *name)
        self.xrt_core.xrtPLKernelOpen.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.xrt_core.xrtPLKernelOpen.restype = ctypes.c_void_p
        
        # xrtKernelClose(xrtKernelHandle khdl)
        self.xrt_core.xrtKernelClose.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtKernelClose.restype = ctypes.c_int
        
        # xrtRunOpen(xrtKernelHandle khdl)
        self.xrt_core.xrtRunOpen.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtRunOpen.restype = ctypes.c_void_p
        
        # xrtRunSetArg(xrtRunHandle rhdl, int index, ...)
        self.xrt_core.xrtRunSetArg.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.xrt_core.xrtRunSetArg.restype = ctypes.c_int
        
        # xrtRunStart(xrtRunHandle rhdl)
        self.xrt_core.xrtRunStart.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtRunStart.restype = ctypes.c_int
        
        # xrtRunWait(xrtRunHandle rhdl)
        self.xrt_core.xrtRunWait.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtRunWait.restype = ctypes.c_int
        
        # xrtRunClose(xrtRunHandle rhdl)
        self.xrt_core.xrtRunClose.argtypes = [ctypes.c_void_p]
        self.xrt_core.xrtRunClose.restype = ctypes.c_int
        
    def open_device(self, device_index: int = 0) -> bool:
        """Open NPU device"""
        try:
            self.device_handle = self.xrt_core.xrtDeviceOpen(device_index)
            if self.device_handle:
                logger.info(f"✅ Opened NPU device {device_index}")
                return True
            else:
                logger.error("Failed to open NPU device")
                return False
        except Exception as e:
            logger.error(f"Error opening device: {e}")
            return False
            
    def allocate_buffer(self, size_bytes: int, name: str, flags: Optional[int] = None) -> Optional[ctypes.c_void_p]:
        """Allocate NPU buffer"""
        try:
            # Use device RAM flag by default for NPU
            if flags is None:
                flags = self.DEFAULT_BO_FLAGS
            
            # Bank: 0 = default bank for NPU
            buffer_handle = self.xrt_core.xrtBOAlloc(
                self.device_handle,
                size_bytes,
                flags,
                0   # bank
            )
            
            if buffer_handle:
                self.buffers[name] = {
                    'handle': buffer_handle,
                    'size': size_bytes
                }
                logger.info(f"✅ Allocated buffer '{name}': {size_bytes} bytes")
                return buffer_handle
            else:
                logger.error(f"Failed to allocate buffer '{name}'")
                return None
                
        except Exception as e:
            logger.error(f"Error allocating buffer: {e}")
            return None
            
    def write_buffer(self, name: str, data: np.ndarray) -> bool:
        """Write data to NPU buffer"""
        try:
            if name not in self.buffers:
                logger.error(f"Buffer '{name}' not found")
                return False
                
            buffer_info = self.buffers[name]
            buffer_handle = buffer_info['handle']
            
            # Convert numpy array to bytes
            data_bytes = data.tobytes()
            
            # Write to buffer
            result = self.xrt_core.xrtBOWrite(
                buffer_handle,
                data_bytes,
                len(data_bytes),
                0  # offset
            )
            
            if result == 0:
                # Sync to device
                sync_result = self.xrt_core.xrtBOSync(
                    buffer_handle,
                    self.XCL_BO_SYNC_BO_TO_DEVICE,
                    len(data_bytes),
                    0   # offset
                )
                
                if sync_result == 0:
                    logger.info(f"✅ Wrote {len(data_bytes)} bytes to buffer '{name}'")
                    return True
                    
            logger.error(f"Failed to write buffer '{name}'")
            return False
            
        except Exception as e:
            logger.error(f"Error writing buffer: {e}")
            return False
            
    def read_buffer(self, name: str, output_array: np.ndarray) -> bool:
        """Read data from NPU buffer"""
        try:
            if name not in self.buffers:
                logger.error(f"Buffer '{name}' not found")
                return False
                
            buffer_info = self.buffers[name]
            buffer_handle = buffer_info['handle']
            
            # Sync from device
            sync_result = self.xrt_core.xrtBOSync(
                buffer_handle,
                self.XCL_BO_SYNC_BO_FROM_DEVICE,
                output_array.nbytes,
                0   # offset
            )
            
            if sync_result == 0:
                # Read from buffer
                result = self.xrt_core.xrtBORead(
                    buffer_handle,
                    output_array.ctypes.data,
                    output_array.nbytes,
                    0  # offset
                )
                
                if result == 0:
                    logger.info(f"✅ Read {output_array.nbytes} bytes from buffer '{name}'")
                    return True
                    
            logger.error(f"Failed to read buffer '{name}'")
            return False
            
        except Exception as e:
            logger.error(f"Error reading buffer: {e}")
            return False
            
    def load_xclbin(self, xclbin_path: str) -> bool:
        """Load XCLBIN binary to NPU"""
        try:
            if not os.path.exists(xclbin_path):
                logger.error(f"XCLBIN not found: {xclbin_path}")
                return False
            
            # Read XCLBIN file
            with open(xclbin_path, 'rb') as f:
                xclbin_data = f.read()
            
            logger.info(f"Loading XCLBIN: {os.path.basename(xclbin_path)} ({len(xclbin_data)} bytes)")
            
            # Load XCLBIN to device
            # Create a ctypes buffer from the data
            xclbin_buffer = ctypes.create_string_buffer(xclbin_data)
            result = self.xrt_core.xrtDeviceLoadXclbin(
                self.device_handle,
                ctypes.cast(xclbin_buffer, ctypes.c_void_p)
            )
            
            if result == 0:
                # Get the UUID of loaded XCLBIN
                self.xclbin_uuid = ctypes.create_string_buffer(16)
                self.xrt_core.xrtDeviceGetXclbinUUID(self.device_handle, self.xclbin_uuid)
                logger.info("✅ XCLBIN loaded successfully")
                return True
            else:
                logger.error(f"Failed to load XCLBIN: error code {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading XCLBIN: {e}")
            return False
    
    def load_kernel(self, kernel_name: str) -> bool:
        """Load kernel from XCLBIN"""
        try:
            if not hasattr(self, 'xclbin_uuid'):
                logger.error("No XCLBIN loaded. Call load_xclbin() first.")
                return False
            
            # Open kernel
            kernel_name_bytes = kernel_name.encode('utf-8')
            self.kernel_handle = self.xrt_core.xrtPLKernelOpen(
                self.device_handle,
                self.xclbin_uuid,
                kernel_name_bytes
            )
            
            if self.kernel_handle:
                logger.info(f"✅ Kernel '{kernel_name}' loaded successfully")
                return True
            else:
                logger.error(f"Failed to load kernel '{kernel_name}'")
                return False
                
        except Exception as e:
            logger.error(f"Error loading kernel: {e}")
            return False
            
    def execute_kernel(self, input_buffer: str, output_buffer: str, 
                      seq_length: int, num_heads: int) -> bool:
        """Execute kernel on NPU"""
        try:
            if not self.kernel_handle:
                logger.error("No kernel loaded")
                return False
            
            logger.info(f"🚀 Executing NPU kernel: seq_len={seq_length}, heads={num_heads}")
            
            # Create run handle
            run_handle = self.xrt_core.xrtRunOpen(self.kernel_handle)
            if not run_handle:
                logger.error("Failed to create run handle")
                return False
            
            # Set kernel arguments
            # For buffer arguments, we need to pass the buffer handle directly
            # Arg 0: input buffer
            input_buf = self.buffers[input_buffer]['handle']
            # xrtRunSetArg with buffer requires passing the handle value
            self.xrt_core.xrtRunSetArg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            self.xrt_core.xrtRunSetArg(run_handle, 0, input_buf)
            
            # Arg 1: output buffer  
            output_buf = self.buffers[output_buffer]['handle']
            self.xrt_core.xrtRunSetArg(run_handle, 1, output_buf)
            
            # For scalar arguments, we need to pass by value
            # Arg 2: sequence length
            self.xrt_core.xrtRunSetArg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
            self.xrt_core.xrtRunSetArg(run_handle, 2, seq_length)
            
            # Arg 3: number of heads
            self.xrt_core.xrtRunSetArg(run_handle, 3, num_heads)
            
            # Start kernel execution
            start_result = self.xrt_core.xrtRunStart(run_handle)
            if start_result != 0:
                logger.error(f"Failed to start kernel: error {start_result}")
                self.xrt_core.xrtRunClose(run_handle)
                return False
            
            # Wait for completion
            wait_result = self.xrt_core.xrtRunWait(run_handle)
            if wait_result != 0:
                logger.error(f"Kernel execution failed: error {wait_result}")
                self.xrt_core.xrtRunClose(run_handle)
                return False
            
            # Close run handle
            self.xrt_core.xrtRunClose(run_handle)
            
            logger.info("✅ NPU kernel execution completed")
            return True
            
        except Exception as e:
            logger.error(f"Error executing kernel: {e}")
            return False
        
    def cleanup(self):
        """Clean up NPU resources"""
        # Free buffers
        for name, buffer_info in self.buffers.items():
            try:
                self.xrt_core.xrtBOFree(buffer_info['handle'])
                logger.info(f"Freed buffer '{name}'")
            except:
                pass
                
        # Close kernel
        if hasattr(self, 'kernel_handle') and self.kernel_handle:
            try:
                self.xrt_core.xrtKernelClose(self.kernel_handle)
                logger.info("Closed kernel")
            except:
                pass
                
        # Close device
        if self.device_handle:
            try:
                self.xrt_core.xrtDeviceClose(self.device_handle)
                logger.info("Closed NPU device")
            except:
                pass


def test_npu_executor():
    """Test NPU kernel executor"""
    
    logger.info("🧪 Testing NPU Kernel Executor...")
    
    executor = NPUKernelExecutor()
    
    # Open device
    if not executor.open_device(0):
        logger.error("Failed to open NPU device")
        return
    
    # Try our compiled XCLBIN first, then fall back to system ones
    xclbin_candidates = [
        "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels/advanced_attention_phoenix.xclbin",
        "/opt/xilinx/xrt/amdxdna/bins/1502_00/validate.xclbin"
    ]
    
    xclbin_path = None
    for candidate in xclbin_candidates:
        if os.path.exists(candidate):
            xclbin_path = candidate
            logger.info(f"Using XCLBIN: {candidate}")
            break
    
    if xclbin_path is None:
        logger.error("No XCLBIN found")
        return
        
    # Load XCLBIN
    if not executor.load_xclbin(xclbin_path):
        logger.error("Failed to load XCLBIN")
        return
    
    # For gemm.xclbin, the kernel is usually named "gemm" or "dpu"
    # For validate.xclbin, it might be "validate" or "dpu"
    kernel_name = "dpu" if "gemm" in xclbin_path else "validate"
    
    if not executor.load_kernel(kernel_name):
        # Try alternative names
        for alt_name in ["gemm", "dpu", "validate", "kernel"]:
            logger.info(f"Trying kernel name: {alt_name}")
            if executor.load_kernel(alt_name):
                kernel_name = alt_name
                break
        else:
            logger.error("Failed to load any kernel")
            return
    
    # Test with small buffers first
    test_size = 1024  # 1KB for initial test
    
    # Allocate buffers
    input_buffer = executor.allocate_buffer(test_size, "input")
    output_buffer = executor.allocate_buffer(test_size, "output")
    
    if input_buffer and output_buffer:
        # Create test data
        test_data = np.random.randn(test_size // 4).astype(np.float32)
        
        # Write to NPU
        if executor.write_buffer("input", test_data):
            logger.info("✅ Data written to NPU")
            
            # Execute kernel
            if executor.execute_kernel("input", "output", 16, 16):
                # Read results
                output_data = np.zeros(test_size // 4, dtype=np.float32)
                if executor.read_buffer("output", output_data):
                    logger.info("✅ Results read from NPU")
                    logger.info(f"Output data shape: {output_data.shape}")
                    logger.info(f"Output mean: {output_data.mean():.4f}, std: {output_data.std():.4f}")
        
    # Cleanup
    executor.cleanup()
    
    logger.info("✅ NPU executor test complete")


if __name__ == "__main__":
    test_npu_executor()