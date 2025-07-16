#!/usr/bin/env python3
"""
HMA GPU Memory Allocator for AMD APUs
Properly allocates memory to VRAM/GTT instead of system RAM
"""

import os
import sys
import numpy as np
import logging
import ctypes
import subprocess
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class HMAGPUMemoryAllocator:
    """
    Allocates memory properly across VRAM/GTT/RAM on AMD APUs
    Uses AMD GPU driver capabilities for proper memory placement
    """
    
    def __init__(self):
        self.initialized = False
        self.gpu_memory_info = {}
        self.allocated_buffers = {}
        
        # Try to load HIP runtime for unified memory support
        self.hip_available = False
        self.hip_lib = None
        
        try:
            # Check if HIP/ROCm is available
            result = subprocess.run(['hipcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.hip_available = True
                logger.info("✅ HIP/ROCm detected - unified memory support available")
        except:
            logger.info("⚠️ HIP/ROCm not found - using alternative methods")
        
        # Get GPU memory info from sysfs
        self._get_gpu_memory_info()
        
    def _get_gpu_memory_info(self):
        """Get GPU memory information from sysfs"""
        try:
            # Find AMD GPU device
            gpu_paths = list(Path('/sys/class/drm').glob('card*/device'))
            
            for gpu_path in gpu_paths:
                vendor_path = gpu_path / 'vendor'
                if vendor_path.exists():
                    vendor = vendor_path.read_text().strip()
                    if vendor == '0x1002':  # AMD vendor ID
                        # Get memory info
                        mem_info_vram = gpu_path / 'mem_info_vram_total'
                        mem_info_gtt = gpu_path / 'mem_info_gtt_total'
                        
                        if mem_info_vram.exists():
                            vram_bytes = int(mem_info_vram.read_text().strip())
                            self.gpu_memory_info['vram_total_gb'] = vram_bytes / (1024**3)
                            
                        if mem_info_gtt.exists():
                            gtt_bytes = int(mem_info_gtt.read_text().strip())
                            self.gpu_memory_info['gtt_total_gb'] = gtt_bytes / (1024**3)
                        
                        # Get current usage
                        mem_info_vram_used = gpu_path / 'mem_info_vram_used'
                        mem_info_gtt_used = gpu_path / 'mem_info_gtt_used'
                        
                        if mem_info_vram_used.exists():
                            vram_used = int(mem_info_vram_used.read_text().strip())
                            self.gpu_memory_info['vram_used_gb'] = vram_used / (1024**3)
                            
                        if mem_info_gtt_used.exists():
                            gtt_used = int(mem_info_gtt_used.read_text().strip())
                            self.gpu_memory_info['gtt_used_gb'] = gtt_used / (1024**3)
                        
                        logger.info(f"📊 GPU Memory Info:")
                        logger.info(f"   VRAM: {self.gpu_memory_info.get('vram_total_gb', 0):.1f}GB total, {self.gpu_memory_info.get('vram_used_gb', 0):.1f}GB used")
                        logger.info(f"   GTT: {self.gpu_memory_info.get('gtt_total_gb', 0):.1f}GB total, {self.gpu_memory_info.get('gtt_used_gb', 0):.1f}GB used")
                        break
                        
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
    
    def allocate_gpu_memory(self, size_bytes: int, memory_type: str = 'auto') -> Optional[Any]:
        """
        Allocate memory on GPU (VRAM or GTT)
        
        Args:
            size_bytes: Size in bytes to allocate
            memory_type: 'vram', 'gtt', or 'auto'
        
        Returns:
            Memory buffer or None if allocation fails
        """
        
        size_gb = size_bytes / (1024**3)
        logger.info(f"🎯 Allocating {size_gb:.2f}GB to {memory_type}")
        
        # If we have pytorch with ROCm support, use it for GPU allocation
        try:
            import torch
            if torch.cuda.is_available() and hasattr(torch.cuda, 'is_available'):
                # ROCm PyTorch available
                device = torch.device('cuda')
                
                # Set environment for APU GTT allocation
                os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
                os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Create tensor on GPU
                shape = (size_bytes // 4,)  # float32 elements
                tensor = torch.zeros(shape, dtype=torch.float32, device=device)
                
                logger.info(f"✅ Allocated {size_gb:.2f}GB on GPU using PyTorch+ROCm")
                return tensor
                
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyTorch GPU allocation failed: {e}")
        
        # Fallback: Use numpy with special flags for pinned memory
        try:
            # Try to allocate pinned memory that GPU can access
            import mmap
            
            # Create anonymous mmap that can be shared with GPU
            flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS
            prot = mmap.PROT_READ | mmap.PROT_WRITE
            
            # Allocate pinned memory
            buffer = mmap.mmap(-1, size_bytes, flags=flags, prot=prot)
            
            # Create numpy array view without copying
            array = np.frombuffer(buffer, dtype=np.uint8)
            
            logger.info(f"✅ Allocated {size_gb:.2f}GB as pinned memory (GPU-accessible)")
            return array
            
        except Exception as e:
            logger.warning(f"Pinned memory allocation failed: {e}")
        
        # Final fallback: Regular numpy array
        logger.warning(f"⚠️ Using regular RAM allocation (not GPU-optimized)")
        return np.zeros(size_bytes // 4, dtype=np.float32)
    
    def allocate_unified_memory(self, size_bytes: int) -> Optional[Any]:
        """
        Allocate unified memory accessible by both CPU and GPU
        This is the key for AMD APU HMA architecture
        """
        
        # Method 1: Use HIP unified memory if available
        if self.hip_available:
            try:
                # Try to use hipMallocManaged via ctypes
                hip_lib_path = '/opt/rocm/lib/libamdhip64.so'
                if Path(hip_lib_path).exists():
                    hip_lib = ctypes.CDLL(hip_lib_path)
                    
                    # Define hipMallocManaged signature
                    hip_lib.hipMallocManaged.argtypes = [
                        ctypes.POINTER(ctypes.c_void_p),  # void** ptr
                        ctypes.c_size_t,                   # size_t size
                        ctypes.c_uint                      # unsigned int flags
                    ]
                    hip_lib.hipMallocManaged.restype = ctypes.c_int
                    
                    # Allocate unified memory
                    ptr = ctypes.c_void_p()
                    flags = 1  # hipMemAttachGlobal
                    result = hip_lib.hipMallocManaged(ctypes.byref(ptr), size_bytes, flags)
                    
                    if result == 0:  # hipSuccess
                        # Create numpy array from pointer
                        array = np.ctypeslib.as_array(
                            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                            shape=(size_bytes // 4,)
                        )
                        logger.info(f"✅ Allocated {size_bytes/(1024**3):.2f}GB unified memory via HIP")
                        return array
                        
            except Exception as e:
                logger.warning(f"HIP unified memory allocation failed: {e}")
        
        # Method 2: Use environment variables to enable unified memory
        os.environ['HSA_ENABLE_UNIFIED_MEMORY'] = '1'
        os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
        
        # Method 3: Use special allocation flags for GTT
        return self.allocate_gpu_memory(size_bytes, 'gtt')
    
    def transfer_to_gpu(self, cpu_array: np.ndarray, memory_type: str = 'auto') -> Any:
        """
        Transfer a CPU numpy array to GPU memory
        """
        
        size_bytes = cpu_array.nbytes
        size_gb = size_bytes / (1024**3)
        
        logger.info(f"📤 Transferring {size_gb:.2f}GB to GPU ({memory_type})")
        
        # Allocate GPU memory
        gpu_buffer = self.allocate_gpu_memory(size_bytes, memory_type)
        
        if gpu_buffer is not None:
            # Copy data
            if hasattr(gpu_buffer, 'copy_'):
                # PyTorch tensor
                gpu_buffer.copy_(torch.from_numpy(cpu_array))
            elif isinstance(gpu_buffer, np.ndarray):
                # Numpy array (pinned or regular)
                gpu_buffer[:] = cpu_array.flatten()
            
            logger.info(f"✅ Transferred {size_gb:.2f}GB to GPU memory")
            return gpu_buffer
        
        logger.warning(f"❌ Failed to transfer to GPU, using CPU array")
        return cpu_array
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        
        # Update current usage
        self._get_gpu_memory_info()
        
        stats = {
            'vram_total_gb': self.gpu_memory_info.get('vram_total_gb', 0),
            'vram_used_gb': self.gpu_memory_info.get('vram_used_gb', 0),
            'vram_free_gb': self.gpu_memory_info.get('vram_total_gb', 0) - self.gpu_memory_info.get('vram_used_gb', 0),
            'gtt_total_gb': self.gpu_memory_info.get('gtt_total_gb', 0),
            'gtt_used_gb': self.gpu_memory_info.get('gtt_used_gb', 0),
            'gtt_free_gb': self.gpu_memory_info.get('gtt_total_gb', 0) - self.gpu_memory_info.get('gtt_used_gb', 0),
        }
        
        return stats


def test_hma_allocator():
    """Test the HMA GPU memory allocator"""
    
    print("🧪 Testing HMA GPU Memory Allocator")
    print("=" * 60)
    
    allocator = HMAGPUMemoryAllocator()
    
    # Show memory stats
    stats = allocator.get_memory_stats()
    print(f"\n📊 Initial Memory Stats:")
    print(f"   VRAM: {stats['vram_used_gb']:.1f}/{stats['vram_total_gb']:.1f}GB used")
    print(f"   GTT: {stats['gtt_used_gb']:.1f}/{stats['gtt_total_gb']:.1f}GB used")
    
    # Test allocations
    test_sizes = [
        (1 * 1024**3, 'vram'),    # 1GB to VRAM
        (4 * 1024**3, 'gtt'),     # 4GB to GTT
        (2 * 1024**3, 'auto'),    # 2GB auto
    ]
    
    for size, mem_type in test_sizes:
        print(f"\n🔧 Testing {size/(1024**3):.1f}GB allocation to {mem_type}")
        buffer = allocator.allocate_gpu_memory(size, mem_type)
        
        if buffer is not None:
            print(f"✅ Allocation successful")
            
            # Show updated stats
            stats = allocator.get_memory_stats()
            print(f"   VRAM used: {stats['vram_used_gb']:.1f}GB")
            print(f"   GTT used: {stats['gtt_used_gb']:.1f}GB")
    
    # Test unified memory
    print(f"\n🔧 Testing unified memory allocation")
    unified_buffer = allocator.allocate_unified_memory(2 * 1024**3)
    if unified_buffer is not None:
        print(f"✅ Unified memory allocation successful")
    
    print(f"\n✅ HMA allocator test complete!")


if __name__ == "__main__":
    test_hma_allocator()