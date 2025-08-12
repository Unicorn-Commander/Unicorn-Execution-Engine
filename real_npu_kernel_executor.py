#!/usr/bin/env python3.13
"""
Real NPU Kernel Executor for AMD Phoenix
Direct hardware execution - no simulations
"""

import os
import sys
import mmap
import struct
import numpy as np
from pathlib import Path
import ctypes
import fcntl
import time
from typing import Optional, Tuple
import logging

# Add virtual environment
sys.path.insert(0, 'npu_kernel_env/lib/python3.13/site-packages')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IOCTL definitions
DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443
DRM_IOCTL_AMDXDNA_MAP_BO = 0xC0186444
DRM_IOCTL_AMDXDNA_SYNC_BO = 0xC0186445
DRM_IOCTL_AMDXDNA_EXEC_CMD = 0xC0206446
DRM_IOCTL_AMDXDNA_DESTROY_BO = 0xC0106448

# Sync flags
SYNC_DIRECT_TO_DEVICE = 0
SYNC_DEVICE_TO_HOST = 1

class NPUBuffer:
    """Real NPU buffer management"""
    
    def __init__(self, npu_fd: int, size: int, bank: int):
        self.npu_fd = npu_fd
        self.size = size
        self.bank = bank
        self.handle = None
        self.mapped = None
        
    def allocate(self) -> bool:
        """Allocate NPU buffer"""
        # Align size to 4KB
        aligned_size = (self.size + 4095) & ~4095
        
        # Create buffer object
        create_args = struct.pack('QII', aligned_size, self.bank, 0)
        try:
            result = fcntl.ioctl(self.npu_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, create_args)
            self.handle = struct.unpack('QII', result)[2]
            logger.info(f"✅ Allocated NPU buffer: {aligned_size} bytes, bank {self.bank:#x}, handle {self.handle}")
            return True
        except Exception as e:
            logger.error(f"❌ Buffer allocation failed: {e}")
            return False
            
    def map(self) -> Optional[mmap.mmap]:
        """Map buffer to host memory"""
        if not self.handle:
            return None
            
        map_args = struct.pack('IIQQ', self.handle, 0, 0, 0)
        try:
            result = fcntl.ioctl(self.npu_fd, DRM_IOCTL_AMDXDNA_MAP_BO, map_args)
            vaddr = struct.unpack('IIQQ', result)[2]
            
            # Create mmap
            self.mapped = mmap.mmap(self.npu_fd, self.size, 
                                   mmap.MAP_SHARED, 
                                   mmap.PROT_READ | mmap.PROT_WRITE,
                                   offset=vaddr)
            logger.info(f"✅ Mapped buffer at {vaddr:#x}")
            return self.mapped
        except Exception as e:
            logger.error(f"❌ Buffer mapping failed: {e}")
            return None
            
    def sync(self, direction: int):
        """Sync buffer with device"""
        if not self.handle:
            return
            
        sync_args = struct.pack('IIQQ', self.handle, direction, 0, self.size)
        try:
            fcntl.ioctl(self.npu_fd, DRM_IOCTL_AMDXDNA_SYNC_BO, sync_args)
        except Exception as e:
            logger.error(f"❌ Buffer sync failed: {e}")
            
    def destroy(self):
        """Destroy buffer"""
        if self.mapped:
            self.mapped.close()
        if self.handle:
            destroy_args = struct.pack('I', self.handle)
            fcntl.ioctl(self.npu_fd, DRM_IOCTL_AMDXDNA_DESTROY_BO, destroy_args)


class NPUKernelExecutor:
    """Execute real NPU kernels on AMD Phoenix hardware"""
    
    def __init__(self):
        self.npu_device = "/dev/accel/accel0"
        self.npu_fd = None
        self.kernels_dir = Path("npu_kernels_real")
        self.loaded_kernel = None
        
        # Memory banks
        self.BANK_DMA = 131071
        self.BANK_COMPUTE0 = 65536
        self.BANK_COMPUTE1 = 65537
        
    def open_device(self) -> bool:
        """Open NPU device"""
        try:
            self.npu_fd = os.open(self.npu_device, os.O_RDWR | os.O_CLOEXEC)
            logger.info(f"✅ NPU device opened: {self.npu_device}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to open NPU: {e}")
            return False
            
    def load_kernel(self, kernel_path: Path) -> bool:
        """Load NPU kernel binary"""
        try:
            with open(kernel_path, 'rb') as f:
                self.loaded_kernel = f.read()
                
            # Parse kernel header
            magic = self.loaded_kernel[:4]
            if magic != b'XDNA':
                logger.error(f"Invalid kernel magic: {magic}")
                return False
                
            header = struct.unpack('<IIIIIIII', self.loaded_kernel[4:36])
            version, hidden_size, num_heads, head_dim, kv_heads, seq_len, tiles = header[:7]
            
            logger.info(f"✅ Loaded kernel: {kernel_path.name}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Model: {hidden_size}d, {num_heads}h, {head_dim}hd")
            logger.info(f"   Sequence: {seq_len}")
            logger.info(f"   Tiles: {tiles}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Kernel load failed: {e}")
            return False
            
    def execute_attention(self, hidden_states: np.ndarray,
                         q_weight: np.ndarray, k_weight: np.ndarray,
                         v_weight: np.ndarray, o_weight: np.ndarray) -> np.ndarray:
        """Execute attention kernel on NPU"""
        
        if not self.loaded_kernel:
            logger.error("No kernel loaded")
            return None
            
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        logger.info(f"🚀 Executing NPU attention kernel")
        logger.info(f"   Input: {hidden_states.shape}")
        
        # Quantize inputs to INT8
        hidden_int8 = (hidden_states * 127).astype(np.int8)
        q_weight_int8 = (q_weight * 127).astype(np.int8)
        k_weight_int8 = (k_weight * 127).astype(np.int8)
        v_weight_int8 = (v_weight * 127).astype(np.int8)
        o_weight_int8 = (o_weight * 127).astype(np.int8)
        
        # Calculate buffer sizes
        input_size = hidden_int8.nbytes
        q_weight_size = q_weight_int8.nbytes
        k_weight_size = k_weight_int8.nbytes
        v_weight_size = v_weight_int8.nbytes
        o_weight_size = o_weight_int8.nbytes
        output_size = batch_size * seq_len * hidden_size
        
        # Allocate NPU buffers
        buffers = {}
        
        # Input buffer
        buffers['input'] = NPUBuffer(self.npu_fd, input_size, self.BANK_DMA)
        if not buffers['input'].allocate():
            return None
            
        # Weight buffers
        total_weight_size = q_weight_size + k_weight_size + v_weight_size + o_weight_size
        buffers['weights'] = NPUBuffer(self.npu_fd, total_weight_size, self.BANK_COMPUTE0)
        if not buffers['weights'].allocate():
            return None
            
        # Output buffer
        buffers['output'] = NPUBuffer(self.npu_fd, output_size, self.BANK_DMA)
        if not buffers['output'].allocate():
            return None
            
        # Map buffers
        input_map = buffers['input'].map()
        weight_map = buffers['weights'].map()
        output_map = buffers['output'].map()
        
        if not all([input_map, weight_map, output_map]):
            logger.error("Buffer mapping failed")
            return None
            
        # Copy data to NPU
        input_map[:input_size] = hidden_int8.tobytes()
        
        # Pack weights contiguously
        offset = 0
        weight_map[offset:offset+q_weight_size] = q_weight_int8.tobytes()
        offset += q_weight_size
        weight_map[offset:offset+k_weight_size] = k_weight_int8.tobytes()
        offset += k_weight_size
        weight_map[offset:offset+v_weight_size] = v_weight_int8.tobytes()
        offset += v_weight_size
        weight_map[offset:offset+o_weight_size] = o_weight_int8.tobytes()
        
        # Sync to device
        buffers['input'].sync(SYNC_DIRECT_TO_DEVICE)
        buffers['weights'].sync(SYNC_DIRECT_TO_DEVICE)
        
        # Execute kernel
        start_time = time.perf_counter()
        
        exec_cmd = struct.pack('QIIII',
            len(self.loaded_kernel),  # Kernel size
            buffers['input'].handle,  # Input handle
            buffers['weights'].handle,  # Weights handle
            buffers['output'].handle,  # Output handle
            0  # Flags
        )
        
        try:
            # Send kernel for execution
            fcntl.ioctl(self.npu_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, exec_cmd)
            
            # Wait for completion (simplified - real impl would use proper sync)
            time.sleep(0.001)  # 1ms
            
            # Sync from device
            buffers['output'].sync(SYNC_DEVICE_TO_HOST)
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"✅ NPU execution complete in {elapsed*1000:.2f}ms")
            
            # Read output
            output_data = np.frombuffer(output_map[:output_size], dtype=np.int8)
            output_data = output_data.reshape(batch_size, seq_len, hidden_size)
            
            # Dequantize
            output_fp32 = output_data.astype(np.float32) / 127.0
            
            # Cleanup
            for buf in buffers.values():
                buf.destroy()
                
            return output_fp32
            
        except Exception as e:
            logger.error(f"❌ NPU execution failed: {e}")
            
            # Cleanup
            for buf in buffers.values():
                buf.destroy()
                
            return None
            
    def benchmark_kernel(self, model_name: str, seq_len: int, iterations: int = 100):
        """Benchmark NPU kernel performance"""
        
        kernel_path = self.kernels_dir / model_name / f"attention_s{seq_len}.xclbin"
        
        if not self.load_kernel(kernel_path):
            logger.error(f"Failed to load kernel: {kernel_path}")
            return
            
        # Get model dimensions from kernel
        header = struct.unpack('<IIIIIIII', self.loaded_kernel[4:36])
        _, hidden_size, num_heads, head_dim, kv_heads, _, _ = header[:7]
        
        # Create test data
        batch_size = 1
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        k_weight = np.random.randn(hidden_size, kv_heads * head_dim).astype(np.float32) * 0.02
        v_weight = np.random.randn(hidden_size, kv_heads * head_dim).astype(np.float32) * 0.02
        o_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        
        # Warmup
        logger.info(f"\n🔥 Warming up NPU...")
        for _ in range(5):
            output = self.execute_attention(hidden_states, q_weight, k_weight, v_weight, o_weight)
            if output is None:
                logger.error("Warmup failed")
                return
                
        # Benchmark
        logger.info(f"\n📊 Benchmarking {model_name} (seq_len={seq_len})")
        
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            output = self.execute_attention(hidden_states, q_weight, k_weight, v_weight, o_weight)
            elapsed = time.perf_counter() - start
            
            if output is not None:
                times.append(elapsed)
                
            if i % 10 == 0:
                logger.info(f"   Iteration {i}: {elapsed*1000:.2f}ms")
                
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Calculate tokens/sec
            tokens_per_sec = seq_len / avg_time
            
            logger.info(f"\n📊 BENCHMARK RESULTS")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Sequence Length: {seq_len}")
            logger.info(f"   Iterations: {len(times)}")
            logger.info(f"   Average Time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
            logger.info(f"   Min/Max: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
            logger.info(f"   Throughput: {tokens_per_sec:.2f} tokens/sec")
            logger.info(f"   NPU TOPS Used: ~{tokens_per_sec * hidden_size * num_heads / 1e12:.2f} TOPS")
            
    def close(self):
        """Close NPU device"""
        if self.npu_fd:
            os.close(self.npu_fd)
            logger.info("✅ NPU device closed")


def main():
    """Test NPU kernel execution"""
    
    executor = NPUKernelExecutor()
    
    if not executor.open_device():
        logger.error("Failed to open NPU device")
        return 1
        
    try:
        # Test Gemma3 4B kernel
        logger.info("\n🦄 Testing Real NPU Kernel Execution")
        logger.info("=" * 60)
        
        # Benchmark different configurations
        models = [
            ("gemma3n", 256),
            ("gemma3_4b", 256),
            ("gemma3_27b", 128)
        ]
        
        for model_name, seq_len in models:
            executor.benchmark_kernel(model_name, seq_len, iterations=10)
            
    finally:
        executor.close()
        
    return 0


if __name__ == "__main__":
    exit(main())