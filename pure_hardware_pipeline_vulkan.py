#!/usr/bin/env python3
"""
Pure Hardware Pipeline with Vulkan GPU Memory
Actually allocates model to VRAM/GTT using Vulkan API
No PyTorch/ROCm dependencies!
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import our components
from real_vulkan_matrix_compute import VulkanMatrixCompute
from npu_attention_kernel_real import NPUAttentionKernelReal
from pure_mmap_loader import MemoryMappedOptimizedLoader
from vulkan_gpu_memory_allocator import VulkanGPUMemoryAllocator

logger = logging.getLogger(__name__)

class PureHardwarePipelineVulkan:
    """Pure hardware inference pipeline with Vulkan GPU memory allocation"""
    
    def __init__(self):
        self.vulkan_engine = None
        self.vulkan_allocator = None
        self.npu_kernel = None
        self.mmap_loader = None
        self.shared_weights = {}
        self.layer_loader = None
        self.initialized = False
        
        # GPU memory allocations
        self.vram_allocations = {}  # layer_idx -> (buffer, memory, tensor)
        self.gtt_allocations = {}   # layer_idx -> (buffer, memory, tensor)
        self.cpu_tensors = {}       # layer_idx -> numpy arrays (fallback)
        
    def initialize(self, model_path: str) -> bool:
        """Initialize pure hardware pipeline with Vulkan GPU memory"""
        try:
            logger.info("🚀 Initializing Pure Hardware Pipeline with Vulkan GPU Memory")
            logger.info("🎮 No PyTorch/ROCm - Direct Vulkan VRAM/GTT allocation")
            
            # Initialize Vulkan GPU allocator
            self.vulkan_allocator = VulkanGPUMemoryAllocator()
            if not self.vulkan_allocator.initialize():
                logger.error("❌ Failed to initialize Vulkan GPU allocator")
                return False
            
            # Show GPU memory stats
            stats = self.vulkan_allocator.get_gpu_memory_stats()
            logger.info(f"📊 GPU Memory Available:")
            logger.info(f"   VRAM: {stats.get('vram_total_mb', 0)/1024:.1f}GB")
            logger.info(f"   GTT: {stats.get('gtt_total_mb', 0)/1024:.1f}GB")
            
            # Initialize Vulkan compute engine
            self.vulkan_engine = VulkanMatrixCompute()
            if not self.vulkan_engine.initialize():
                logger.error("❌ Failed to initialize Vulkan compute engine")
                return False
            logger.info("✅ Vulkan compute engine initialized")
            
            # Initialize NPU kernel
            self.npu_kernel = NPUAttentionKernelReal()
            try:
                if self.npu_kernel.initialize():
                    logger.info("✅ NPU kernel initialized")
                else:
                    logger.warning("⚠️ NPU kernel initialization failed")
            except Exception as e:
                logger.warning(f"⚠️ NPU kernel error: {e}")
            
            # Initialize memory-mapped loader
            self.mmap_loader = MemoryMappedOptimizedLoader(model_path)
            model_info = self.mmap_loader.load_model()
            
            self.shared_weights = model_info.get('shared_weights', {})
            self.layer_loader = model_info.get('layer_loader')
            
            logger.info(f"✅ Memory-mapped loader: {len(self.shared_weights)} shared weights")
            
            # Load model to GPU memory using Vulkan
            self._load_model_to_vulkan_gpu()
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def _load_model_to_vulkan_gpu(self):
        """Load model layers to GPU memory using Vulkan API"""
        
        logger.info("🚀 LOADING MODEL TO GPU USING VULKAN")
        
        # Memory allocation strategy
        vram_budget_mb = 14 * 1024  # 14GB for VRAM (leave 2GB for system)
        gtt_budget_mb = 35 * 1024   # 35GB for GTT (leave some for system)
        
        vram_used_mb = 0
        gtt_used_mb = 0
        cpu_used_mb = 0
        
        logger.info(f"📊 Memory Budget:")
        logger.info(f"   VRAM: {vram_budget_mb/1024:.1f}GB")
        logger.info(f"   GTT: {gtt_budget_mb/1024:.1f}GB")
        
        # Priority layers for VRAM (first 4 and last 4)
        vram_priority_layers = list(range(0, 4)) + list(range(58, 62))
        
        logger.info("📦 Loading model layers to GPU memory...")
        
        for layer_idx in range(62):  # 62 layers total
            try:
                # Load layer weights
                layer_weights = self.layer_loader(layer_idx)
                
                # Calculate layer size
                layer_size_mb = 0
                layer_tensors = {}
                
                for name, weight_info in layer_weights.items():
                    if name.startswith('language_model'):
                        # Get tensor from mmap
                        if weight_info.get('lazy', False) and self.mmap_loader:
                            tensor = self.mmap_loader.get_tensor(weight_info)
                        else:
                            tensor = weight_info.get('tensor')
                        
                        # Convert to numpy
                        if hasattr(tensor, 'numpy'):
                            tensor = tensor.numpy()
                        elif not isinstance(tensor, np.ndarray):
                            tensor = np.array(tensor)
                        
                        layer_tensors[name] = tensor
                        layer_size_mb += tensor.nbytes / (1024**2)
                
                # Decide where to allocate
                if layer_idx in vram_priority_layers and vram_used_mb + layer_size_mb <= vram_budget_mb:
                    # Allocate to VRAM
                    self._allocate_layer_to_vram(layer_idx, layer_tensors)
                    vram_used_mb += layer_size_mb
                    logger.info(f"   ✅ Layer {layer_idx} → VRAM ({layer_size_mb:.1f}MB)")
                    
                elif gtt_used_mb + layer_size_mb <= gtt_budget_mb:
                    # Allocate to GTT
                    self._allocate_layer_to_gtt(layer_idx, layer_tensors)
                    gtt_used_mb += layer_size_mb
                    if layer_idx % 10 == 0:
                        logger.info(f"   ⚡ Layer {layer_idx} → GTT ({layer_size_mb:.1f}MB)")
                    
                else:
                    # Keep in CPU memory
                    self.cpu_tensors[layer_idx] = layer_tensors
                    cpu_used_mb += layer_size_mb
                    logger.warning(f"   ⚠️ Layer {layer_idx} → CPU RAM ({layer_size_mb:.1f}MB)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load layer {layer_idx}: {e}")
        
        # Show final memory distribution
        logger.info(f"🎉 MODEL LOADED TO GPU MEMORY!")
        logger.info(f"   📍 VRAM: {vram_used_mb/1024:.1f}GB ({len(self.vram_allocations)} layers)")
        logger.info(f"   📍 GTT: {gtt_used_mb/1024:.1f}GB ({len(self.gtt_allocations)} layers)")
        logger.info(f"   📍 CPU: {cpu_used_mb/1024:.1f}GB ({len(self.cpu_tensors)} layers)")
        logger.info(f"   📊 Total: {(vram_used_mb + gtt_used_mb + cpu_used_mb)/1024:.1f}GB")
        
        # Verify with system stats
        stats = self.vulkan_allocator.get_gpu_memory_stats()
        logger.info(f"📊 Vulkan Allocator Stats:")
        logger.info(f"   Total allocated: {stats['allocated_mb']/1024:.1f}GB")
        logger.info(f"   Number of allocations: {stats['allocations']}")
    
    def _allocate_layer_to_vram(self, layer_idx: int, layer_tensors: Dict[str, np.ndarray]):
        """Allocate layer to VRAM using Vulkan"""
        
        allocations = {}
        
        for name, tensor in layer_tensors.items():
            # Transfer to GPU VRAM
            buffer, memory = self.vulkan_allocator.transfer_to_gpu(tensor, prefer_device_local=True)
            if buffer and memory:
                allocations[name] = {
                    'buffer': buffer,
                    'memory': memory,
                    'tensor': tensor,  # Keep CPU copy for now
                    'shape': tensor.shape,
                    'dtype': tensor.dtype
                }
        
        self.vram_allocations[layer_idx] = allocations
    
    def _allocate_layer_to_gtt(self, layer_idx: int, layer_tensors: Dict[str, np.ndarray]):
        """Allocate layer to GTT using Vulkan"""
        
        allocations = {}
        
        for name, tensor in layer_tensors.items():
            # Transfer to GPU GTT (host-visible)
            buffer, memory = self.vulkan_allocator.transfer_to_gpu(tensor, prefer_device_local=False)
            if buffer and memory:
                allocations[name] = {
                    'buffer': buffer,
                    'memory': memory,
                    'tensor': tensor,  # Keep CPU copy for now
                    'shape': tensor.shape,
                    'dtype': tensor.dtype
                }
        
        self.gtt_allocations[layer_idx] = allocations
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get layer weights from appropriate memory location"""
        
        # Check in order: VRAM -> GTT -> CPU
        if layer_idx in self.vram_allocations:
            logger.debug(f"⚡ Layer {layer_idx} from VRAM")
            # For now, return the CPU copy
            # TODO: Implement GPU->CPU transfer when needed
            return {name: data['tensor'] for name, data in self.vram_allocations[layer_idx].items()}
            
        elif layer_idx in self.gtt_allocations:
            logger.debug(f"💾 Layer {layer_idx} from GTT")
            return {name: data['tensor'] for name, data in self.gtt_allocations[layer_idx].items()}
            
        elif layer_idx in self.cpu_tensors:
            logger.debug(f"🐌 Layer {layer_idx} from CPU RAM")
            return self.cpu_tensors[layer_idx]
            
        else:
            logger.error(f"❌ Layer {layer_idx} not found!")
            return {}
    
    def cleanup(self):
        """Cleanup hardware resources"""
        if self.vulkan_allocator:
            self.vulkan_allocator.cleanup()
        if self.vulkan_engine:
            self.vulkan_engine.cleanup()
        if self.npu_kernel:
            self.npu_kernel.cleanup()


def test_vulkan_pipeline():
    """Test the Vulkan GPU memory pipeline"""
    
    print("🦄 Testing Pure Hardware Pipeline with Vulkan GPU Memory")
    print("=" * 60)
    
    model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
    
    pipeline = PureHardwarePipelineVulkan()
    
    if not pipeline.initialize(model_path):
        print("❌ Failed to initialize pipeline")
        return False
    
    print("✅ Vulkan pipeline initialized with GPU memory allocation")
    
    # Check memory distribution
    print("\n📊 Model Memory Distribution:")
    print(f"   VRAM layers: {len(pipeline.vram_allocations)}")
    print(f"   GTT layers: {len(pipeline.gtt_allocations)}")
    print(f"   CPU layers: {len(pipeline.cpu_tensors)}")
    
    # Show GPU memory usage
    import subprocess
    print("\n📊 GPU Memory Usage:")
    
    # VRAM
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line and 'GPU[0]' in line:
            vram_used = int(line.split(':')[-1].strip()) / (1024**3)
            print(f"   VRAM Used: {vram_used:.1f}GB")
    
    # GTT
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'gtt'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line and 'GPU[0]' in line:
            gtt_used = int(line.split(':')[-1].strip()) / (1024**3)
            print(f"   GTT Used: {gtt_used:.1f}GB")
    
    pipeline.cleanup()
    print("\n🎉 Vulkan pipeline test completed!")
    return True


if __name__ == "__main__":
    test_vulkan_pipeline()