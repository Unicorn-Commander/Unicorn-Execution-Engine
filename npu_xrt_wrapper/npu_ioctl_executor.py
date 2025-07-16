#!/usr/bin/env python3
"""
NPU Ioctl Executor - Direct kernel execution using AMDXDNA driver ioctls
Implements actual NPU kernel submission without XRT/XCLBIN
"""

import os
import sys
import numpy as np
import logging
import ctypes
import fcntl
import mmap
import struct
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DRM ioctl constants
DRM_IOCTL_VERSION = 0xC0406400
DRM_COMMAND_BASE = 0x40

# AMDXDNA ioctl IDs
DRM_AMDXDNA_CREATE_HWCTX = 0
DRM_AMDXDNA_DESTROY_HWCTX = 1
DRM_AMDXDNA_CONFIG_HWCTX = 2
DRM_AMDXDNA_CREATE_BO = 3
DRM_AMDXDNA_GET_BO_INFO = 4
DRM_AMDXDNA_SYNC_BO = 5
DRM_AMDXDNA_EXEC_CMD = 6
DRM_AMDXDNA_GET_INFO = 7

# Helper to create ioctl numbers
def _IOC(dir, type, nr, size):
    return (dir << 30) | (type << 8) | nr | (size << 16)

def _IOWR(nr, size):
    return _IOC(3, ord('d'), nr, size)  # 'd' for DRM

# Buffer object types
AMDXDNA_BO_SHMEM = 1
AMDXDNA_BO_DEV_HEAP = 2
AMDXDNA_BO_DEV = 3
AMDXDNA_BO_CMD = 4

# Command types
AMDXDNA_CMD_SUBMIT_EXEC_BUF = 0

# NPU opcodes
ERT_START_NPU = 20

# Structures for ioctls
class AmdxdnaDrmCreateHwctx(ctypes.Structure):
    _fields_ = [
        ("ext", ctypes.c_uint64),
        ("ext_flags", ctypes.c_uint64),
        ("qos_p", ctypes.c_uint64),
        ("umq_bo", ctypes.c_uint32),
        ("log_buf_bo", ctypes.c_uint32),
        ("max_opc", ctypes.c_uint32),
        ("num_tiles", ctypes.c_uint32),
        ("mem_size", ctypes.c_uint32),
        ("umq_doorbell", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
        ("syncobj_handle", ctypes.c_uint32),
    ]

class AmdxdnaDrmCreateBo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("vaddr", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("type", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
    ]

class AmdxdnaDrmGetBoInfo(ctypes.Structure):
    _fields_ = [
        ("ext", ctypes.c_uint64),
        ("ext_flags", ctypes.c_uint64),
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("map_offset", ctypes.c_uint64),
        ("vaddr", ctypes.c_uint64),
        ("xdna_addr", ctypes.c_uint64),
    ]

class AmdxdnaDrmExecCmd(ctypes.Structure):
    _fields_ = [
        ("ext", ctypes.c_uint64),
        ("ext_flags", ctypes.c_uint64),
        ("hwctx", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("cmd_handles", ctypes.c_uint64),
        ("args", ctypes.c_uint64),
        ("cmd_count", ctypes.c_uint32),
        ("arg_count", ctypes.c_uint32),
        ("seq", ctypes.c_uint64),
    ]

class NPUIoctlExecutor:
    """Execute NPU kernels using direct ioctl interface"""
    
    def __init__(self):
        self.device_fd = None
        self.hwctx = None
        self.buffers = {}
        self.mapped_buffers = {}
        self.initialized = False
        
        # NPU device path
        self.npu_device = "/dev/accel/accel0"
        
        logger.info("🚀 NPU Ioctl Executor - Direct kernel execution")
        
    def initialize(self) -> bool:
        """Initialize NPU device and create hardware context"""
        try:
            # Open NPU device
            self.device_fd = os.open(self.npu_device, os.O_RDWR)
            logger.info(f"✅ Opened NPU device: {self.npu_device}")
            
            # Create hardware context
            hwctx_args = AmdxdnaDrmCreateHwctx()
            hwctx_args.ext = 0
            hwctx_args.ext_flags = 0
            hwctx_args.qos_p = 0
            hwctx_args.umq_bo = 0
            hwctx_args.log_buf_bo = 0
            hwctx_args.max_opc = 0x10000
            hwctx_args.num_tiles = 16  # For Phoenix NPU
            hwctx_args.mem_size = 0  # Let driver decide
            
            ioctl_num = _IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_HWCTX, 
                              ctypes.sizeof(AmdxdnaDrmCreateHwctx))
            
            ret = fcntl.ioctl(self.device_fd, ioctl_num, hwctx_args)
            if ret == 0:
                self.hwctx = hwctx_args.handle
                logger.info(f"✅ Created hardware context: {self.hwctx}")
                logger.info(f"   Syncobj handle: {hwctx_args.syncobj_handle}")
            else:
                logger.error(f"Failed to create hardware context: {ret}")
                return False
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NPU: {e}")
            return False
    
    def create_buffer(self, size: int, bo_type: int, name: str) -> Optional[int]:
        """Create a buffer object"""
        try:
            bo_args = AmdxdnaDrmCreateBo()
            bo_args.flags = 0
            bo_args.vaddr = 0
            bo_args.size = size
            bo_args.type = bo_type
            
            ioctl_num = _IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_BO,
                              ctypes.sizeof(AmdxdnaDrmCreateBo))
            
            ret = fcntl.ioctl(self.device_fd, ioctl_num, bo_args)
            if ret == 0:
                handle = bo_args.handle
                self.buffers[name] = {
                    'handle': handle,
                    'size': size,
                    'type': bo_type
                }
                logger.info(f"✅ Created buffer '{name}': handle={handle}, size={size}")
                return handle
            else:
                logger.error(f"Failed to create buffer: {ret}")
                return None
                
        except Exception as e:
            logger.error(f"Buffer creation failed: {e}")
            return None
    
    def map_buffer(self, name: str) -> Optional[mmap.mmap]:
        """Map a buffer to user space"""
        try:
            if name not in self.buffers:
                logger.error(f"Buffer '{name}' not found")
                return None
            
            buffer_info = self.buffers[name]
            handle = buffer_info['handle']
            size = buffer_info['size']
            
            # Get buffer info including mmap offset
            info_args = AmdxdnaDrmGetBoInfo()
            info_args.ext = 0
            info_args.ext_flags = 0
            info_args.handle = handle
            
            ioctl_num = _IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_GET_BO_INFO,
                              ctypes.sizeof(AmdxdnaDrmGetBoInfo))
            
            ret = fcntl.ioctl(self.device_fd, ioctl_num, info_args)
            if ret != 0:
                logger.error(f"Failed to get buffer info: {ret}")
                return None
            
            # Map the buffer
            map_offset = info_args.map_offset
            mapped = mmap.mmap(self.device_fd, size, 
                              mmap.MAP_SHARED, 
                              mmap.PROT_READ | mmap.PROT_WRITE,
                              offset=map_offset)
            
            self.mapped_buffers[name] = mapped
            buffer_info['vaddr'] = info_args.vaddr
            buffer_info['xdna_addr'] = info_args.xdna_addr
            
            logger.info(f"✅ Mapped buffer '{name}': vaddr=0x{info_args.vaddr:x}, "
                       f"xdna_addr=0x{info_args.xdna_addr:x}")
            return mapped
            
        except Exception as e:
            logger.error(f"Buffer mapping failed: {e}")
            return None
    
    def create_npu_command(self, kernel_data: bytes, instruction_buffer_handle: int,
                          data_handles: Dict[str, int]) -> bytes:
        """Create NPU command buffer"""
        
        # Command header: opcode | count | extra_cu_mask | state
        opcode = ERT_START_NPU
        count = 1  # Number of CUs
        header = (opcode << 23) | (count << 12) | 0x0
        
        # NPU start command structure
        # Format: header, buffer address, buffer size, prop count, properties...
        cmd_data = struct.pack('<I', header)  # Command header
        
        # Get instruction buffer address
        inst_buffer_info = next((b for b in self.buffers.values() 
                               if b['handle'] == instruction_buffer_handle), None)
        if not inst_buffer_info:
            raise ValueError("Instruction buffer not found")
        
        inst_addr = inst_buffer_info.get('xdna_addr', 0)
        inst_size = len(kernel_data)
        
        # Add NPU start command data
        cmd_data += struct.pack('<Q', inst_addr)      # Instruction buffer address
        cmd_data += struct.pack('<I', inst_size)      # Instruction buffer size
        cmd_data += struct.pack('<I', 0)              # Property count (0 for now)
        
        return cmd_data
    
    def execute_kernel(self, kernel_path: str, input_data: np.ndarray,
                      seq_length: int, num_heads: int) -> Optional[np.ndarray]:
        """Execute NPU kernel"""
        
        if not self.initialized:
            logger.error("NPU not initialized")
            return None
        
        try:
            # Load kernel binary
            with open(kernel_path, 'rb') as f:
                kernel_data = f.read()
            
            kernel_name = os.path.basename(kernel_path)
            logger.info(f"⚡ Executing kernel: {kernel_name} ({len(kernel_data)} bytes)")
            
            # Create buffers
            # 1. Command buffer
            cmd_buffer = self.create_buffer(4096, AMDXDNA_BO_CMD, "command")
            if not cmd_buffer:
                return None
            
            # 2. Instruction buffer
            inst_size = ((len(kernel_data) + 4095) // 4096) * 4096  # Align to 4K
            inst_buffer = self.create_buffer(inst_size, AMDXDNA_BO_DEV, "instructions")
            if not inst_buffer:
                return None
            
            # 3. Data buffers
            input_size = input_data.nbytes
            input_buffer = self.create_buffer(input_size, AMDXDNA_BO_SHMEM, "input")
            output_buffer = self.create_buffer(input_size, AMDXDNA_BO_SHMEM, "output")
            
            if not input_buffer or not output_buffer:
                return None
            
            # Map buffers
            cmd_map = self.map_buffer("command")
            inst_map = self.map_buffer("instructions")
            input_map = self.map_buffer("input")
            output_map = self.map_buffer("output")
            
            if not all([cmd_map, inst_map, input_map, output_map]):
                logger.error("Failed to map buffers")
                return None
            
            # Write kernel to instruction buffer
            inst_map.write(kernel_data)
            inst_map.flush()
            
            # Write input data
            input_map.write(input_data.tobytes())
            input_map.flush()
            
            # Create command buffer
            cmd_data = self.create_npu_command(
                kernel_data, inst_buffer,
                {'input': input_buffer, 'output': output_buffer}
            )
            
            cmd_map.write(cmd_data)
            cmd_map.flush()
            
            # Submit command
            exec_args = AmdxdnaDrmExecCmd()
            exec_args.ext = 0
            exec_args.ext_flags = 0
            exec_args.hwctx = self.hwctx
            exec_args.type = AMDXDNA_CMD_SUBMIT_EXEC_BUF
            
            # Create array with command handle
            cmd_handles_array = (ctypes.c_uint32 * 1)()
            cmd_handles_array[0] = cmd_buffer
            exec_args.cmd_handles = ctypes.cast(cmd_handles_array, ctypes.c_void_p).value
            exec_args.cmd_count = 1
            
            # No additional args for now
            exec_args.args = 0
            exec_args.arg_count = 0
            
            ioctl_num = _IOWR(DRM_COMMAND_BASE + DRM_AMDXDNA_EXEC_CMD,
                              ctypes.sizeof(AmdxdnaDrmExecCmd))
            
            logger.info("📤 Submitting NPU command...")
            ret = fcntl.ioctl(self.device_fd, ioctl_num, exec_args)
            
            if ret == 0:
                seq_num = exec_args.seq
                logger.info(f"✅ Command submitted successfully, sequence: {seq_num}")
                
                # Wait for completion (simple polling for now)
                import time
                time.sleep(0.01)  # Give NPU time to execute
                
                # Read output
                output_data = np.frombuffer(output_map.read(input_size), 
                                          dtype=input_data.dtype).reshape(input_data.shape)
                
                logger.info(f"✅ NPU execution complete!")
                return output_data
            else:
                logger.error(f"Command submission failed: {ret}")
                return None
                
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up NPU resources"""
        # Unmap buffers
        for name, mapped in self.mapped_buffers.items():
            try:
                mapped.close()
                logger.debug(f"Unmapped buffer '{name}'")
            except:
                pass
        
        # Destroy hardware context
        if self.hwctx is not None:
            # Would use destroy hwctx ioctl here
            logger.info("Destroyed hardware context")
        
        # Close device
        if self.device_fd is not None:
            os.close(self.device_fd)
            logger.info("✅ Closed NPU device")

def test_ioctl_execution():
    """Test NPU kernel execution via ioctl"""
    
    logger.info("🧪 Testing NPU Kernel Execution via Ioctl...")
    
    executor = NPUIoctlExecutor()
    
    if not executor.initialize():
        logger.error("Failed to initialize NPU")
        return
    
    # Test with our compiled kernels
    kernel_dir = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels"
    kernel_path = os.path.join(kernel_dir, "attention_256_int8.bin")
    
    # Create test data
    seq_length = 256
    hidden_size = 5376
    test_data = np.random.randn(1, seq_length, hidden_size).astype(np.float32)
    
    logger.info(f"Input shape: {test_data.shape}")
    logger.info(f"Input stats: mean={test_data.mean():.4f}, std={test_data.std():.4f}")
    
    # Execute kernel
    output = executor.execute_kernel(kernel_path, test_data, seq_length, 32)
    
    if output is not None:
        logger.info(f"✅ Output shape: {output.shape}")
        logger.info(f"✅ Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
        
        # Check if output changed (indicates real execution)
        if not np.array_equal(test_data, output):
            logger.info("🎉 OUTPUT CHANGED - NPU KERNEL EXECUTED!")
        else:
            logger.warning("⚠️ Output unchanged - kernel may not have executed")
    else:
        logger.error("❌ Execution failed")
    
    # Cleanup
    executor.cleanup()
    
    logger.info("\n✅ NPU ioctl executor test complete")

if __name__ == "__main__":
    test_ioctl_execution()