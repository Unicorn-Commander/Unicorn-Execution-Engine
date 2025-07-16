#!/usr/bin/env python3
"""
Real GPU Memory Allocator for AMD APUs
Actually allocates memory to VRAM/GTT using PyTorch+ROCm
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class RealGPUMemoryAllocator:
    """
    Actually allocates memory to GPU (VRAM/GTT) using PyTorch+ROCm
    """
    
    def __init__(self):
        # Enable AMD APU optimizations
        os.environ['HSA_ENABLE_UNIFIED_MEMORY'] = '1'
        os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['GPU_MAX_HEAP_SIZE'] = '100'  # Allow GPU to use more memory
        
        self.torch_available = False
        self.device = None
        
        try:
            import torch
            self.torch = torch
            
            if torch.cuda.is_available():
                self.torch_available = True
                self.device = torch.device('cuda:0')
                
                # Get GPU info
                props = torch.cuda.get_device_properties(0)
                self.gpu_name = torch.cuda.get_device_name(0)
                self.total_memory_gb = props.total_memory / (1024**3)
                
                logger.info(f"✅ PyTorch+ROCm initialized")
                logger.info(f"   GPU: {self.gpu_name}")
                logger.info(f"   Total Memory: {self.total_memory_gb:.1f}GB")
                
                # Enable memory expansion for APUs
                torch.cuda.set_per_process_memory_fraction(1.0)
                
            else:
                logger.error("❌ ROCm not available in PyTorch")
                
        except ImportError:
            logger.error("❌ PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
    
    def allocate_to_vram(self, size_gb: float, dtype=None) -> Optional[Any]:
        """Actually allocate memory in VRAM"""
        if not self.torch_available:
            logger.error("PyTorch+ROCm not available")
            return None
        
        if dtype is None:
            dtype = self.torch.float32
        
        # Calculate number of elements
        bytes_per_element = 4 if dtype == self.torch.float32 else 2
        num_elements = int(size_gb * 1024**3 / bytes_per_element)
        
        try:
            # This ACTUALLY allocates on GPU!
            tensor = self.torch.zeros(num_elements, dtype=dtype, device=self.device)
            
            # Verify allocation
            allocated_gb = self.torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = self.torch.cuda.memory_reserved() / (1024**3)
            
            logger.info(f"✅ Allocated {size_gb:.1f}GB to VRAM")
            logger.info(f"   Allocated: {allocated_gb:.1f}GB")
            logger.info(f"   Reserved: {reserved_gb:.1f}GB")
            
            return tensor
            
        except Exception as e:
            logger.error(f"❌ VRAM allocation failed: {e}")
            return None
    
    def transfer_to_gpu(self, cpu_array: np.ndarray) -> Optional[Any]:
        """Transfer numpy array to GPU memory"""
        if not self.torch_available:
            return None
        
        try:
            # Convert numpy to torch tensor on GPU
            gpu_tensor = self.torch.from_numpy(cpu_array).to(self.device)
            
            size_gb = cpu_array.nbytes / (1024**3)
            logger.info(f"✅ Transferred {size_gb:.1f}GB to GPU")
            
            return gpu_tensor
            
        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            return None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get real GPU memory statistics"""
        if not self.torch_available:
            return {}
        
        return {
            'allocated_gb': self.torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': self.torch.cuda.memory_reserved() / (1024**3),
            'free_gb': (self.torch.cuda.get_device_properties(0).total_memory - 
                       self.torch.cuda.memory_allocated()) / (1024**3),
            'total_gb': self.torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    def load_model_to_gpu(self, model_path: str) -> Dict[str, Any]:
        """Load model directly to GPU memory"""
        if not self.torch_available:
            logger.error("PyTorch+ROCm required for GPU loading")
            return {}
        
        logger.info(f"🚀 Loading model to GPU: {model_path}")
        
        # Import safetensors
        try:
            from safetensors import safe_open
        except ImportError:
            logger.error("Install safetensors: pip install safetensors")
            return {}
        
        gpu_tensors = {}
        total_size_gb = 0
        
        # Find all model files
        model_files = list(Path(model_path).glob("*.safetensors"))
        
        for model_file in model_files:
            logger.info(f"📂 Loading {model_file.name}")
            
            with safe_open(model_file, framework="pt", device="cuda:0") as f:
                for key in f.keys():
                    # Load tensor directly to GPU
                    tensor = f.get_tensor(key)
                    gpu_tensors[key] = tensor
                    
                    size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
                    total_size_gb += size_gb
                    
                    if len(gpu_tensors) % 100 == 0:
                        stats = self.get_memory_stats()
                        logger.info(f"   Loaded {len(gpu_tensors)} tensors, "
                                  f"GPU memory: {stats['allocated_gb']:.1f}GB / {stats['total_gb']:.1f}GB")
        
        logger.info(f"✅ Loaded {len(gpu_tensors)} tensors ({total_size_gb:.1f}GB) to GPU")
        return gpu_tensors


def test_real_gpu_allocation():
    """Test real GPU memory allocation"""
    print("🧪 Testing Real GPU Memory Allocation")
    print("=" * 60)
    
    allocator = RealGPUMemoryAllocator()
    
    if not allocator.torch_available:
        print("❌ PyTorch+ROCm not available. Install with:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        return
    
    # Show initial state
    stats = allocator.get_memory_stats()
    print(f"\n📊 Initial GPU Memory:")
    print(f"   Total: {stats['total_gb']:.1f}GB")
    print(f"   Free: {stats['free_gb']:.1f}GB")
    
    # Test VRAM allocation
    print(f"\n🔧 Allocating 4GB to VRAM...")
    tensor = allocator.allocate_to_vram(4.0)
    
    if tensor is not None:
        stats = allocator.get_memory_stats()
        print(f"📊 After allocation:")
        print(f"   Allocated: {stats['allocated_gb']:.1f}GB")
        print(f"   Free: {stats['free_gb']:.1f}GB")
        
        # Verify with rocm-smi
        import subprocess
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                              capture_output=True, text=True)
        print(f"\n📊 rocm-smi reports:")
        for line in result.stdout.split('\n'):
            if 'Used' in line and 'GPU' in line:
                print(f"   {line.strip()}")
    
    print("\n✅ Real GPU allocation test complete!")


if __name__ == "__main__":
    test_real_gpu_allocation()