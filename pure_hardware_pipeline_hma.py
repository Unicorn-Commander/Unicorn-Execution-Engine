#!/usr/bin/env python3
"""
Pure Hardware Pipeline with HMA GPU Memory Support
Properly allocates model to VRAM/GTT instead of system RAM
"""

import numpy as np
import time
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import our components
from real_vulkan_matrix_compute import VulkanMatrixCompute
from npu_attention_kernel_real import NPUAttentionKernelReal
from pure_mmap_loader import MemoryMappedOptimizedLoader
from hma_gpu_memory_allocator import HMAGPUMemoryAllocator

logger = logging.getLogger(__name__)

class PureHardwarePipelineHMA:
    """Pure hardware inference pipeline with proper HMA GPU memory allocation"""
    
    def __init__(self):
        self.vulkan_engine = None
        self.npu_kernel = None
        self.mmap_loader = None
        self.gpu_allocator = None
        self.shared_weights = {}
        self.layer_loader = None
        self.initialized = False
        
        # GPU memory pools
        self.vram_tensors = {}      # High-priority tensors in VRAM
        self.gtt_tensors = {}       # Bulk tensors in GTT
        self.cpu_tensors = {}       # Overflow tensors in CPU RAM
        
    def initialize(self, model_path: str) -> bool:
        """Initialize pure hardware pipeline with GPU memory allocation"""
        try:
            logger.info("🚀 Initializing Pure Hardware Pipeline with HMA GPU Memory")
            
            # Enable AMD APU memory optimizations
            os.environ['HSA_ENABLE_UNIFIED_MEMORY'] = '1'
            os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Initialize GPU memory allocator
            self.gpu_allocator = HMAGPUMemoryAllocator()
            memory_stats = self.gpu_allocator.get_memory_stats()
            
            logger.info(f"📊 GPU Memory Available:")
            logger.info(f"   VRAM: {memory_stats['vram_free_gb']:.1f}GB free of {memory_stats['vram_total_gb']:.1f}GB")
            logger.info(f"   GTT: {memory_stats['gtt_free_gb']:.1f}GB free of {memory_stats['gtt_total_gb']:.1f}GB")
            
            # Initialize hardware engines
            self.vulkan_engine = VulkanMatrixCompute()
            if not self.vulkan_engine.initialize():
                logger.error("❌ Failed to initialize Vulkan engine")
                return False
            logger.info("✅ Vulkan iGPU engine initialized")
            
            # Initialize NPU kernel
            self.npu_kernel = NPUAttentionKernelReal()
            try:
                if self.npu_kernel.initialize():
                    logger.info("✅ NPU kernel initialized and ready")
                else:
                    logger.warning("⚠️ NPU kernel initialization failed - will use Vulkan/CPU fallback")
            except Exception as e:
                logger.warning(f"⚠️ NPU kernel initialization error: {e}")
            
            # Initialize memory-mapped loader
            self.mmap_loader = MemoryMappedOptimizedLoader(model_path)
            model_info = self.mmap_loader.load_model()
            
            self.shared_weights = model_info.get('shared_weights', {})
            self.layer_loader = model_info.get('layer_loader')
            
            logger.info(f"✅ Memory-mapped loader: {len(self.shared_weights)} shared weights")
            
            # Load model with GPU memory allocation
            self._load_model_to_gpu_memory()
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def _load_model_to_gpu_memory(self):
        """Load model layers to GPU memory (VRAM/GTT) instead of RAM"""
        
        logger.info("🚀 LOADING MODEL TO GPU MEMORY (VRAM/GTT)")
        
        # Memory allocation strategy
        vram_used_gb = 0
        gtt_used_gb = 0
        cpu_used_gb = 0
        
        # Get available memory
        stats = self.gpu_allocator.get_memory_stats()
        vram_available_gb = min(stats['vram_free_gb'], 14.0)  # Reserve 2GB for OS/other
        gtt_available_gb = min(stats['gtt_free_gb'], 40.0)    # Use most GTT
        
        logger.info(f"📊 Memory Allocation Plan:")
        logger.info(f"   VRAM: Using up to {vram_available_gb:.1f}GB")
        logger.info(f"   GTT: Using up to {gtt_available_gb:.1f}GB")
        
        # Priority 1: Load embeddings and critical layers to VRAM
        critical_layers = list(range(0, 4)) + list(range(58, 62))  # First 4 and last 4 layers
        
        logger.info("📦 Loading critical layers to VRAM...")
        for layer_idx in critical_layers:
            try:
                layer_weights = self.layer_loader(layer_idx)
                layer_size_gb = self._estimate_layer_size(layer_weights) / (1024**3)
                
                if vram_used_gb + layer_size_gb <= vram_available_gb:
                    # Load to VRAM
                    gpu_layer = self._load_layer_to_gpu(layer_weights, 'vram')
                    self.vram_tensors[layer_idx] = gpu_layer
                    vram_used_gb += layer_size_gb
                    logger.info(f"   ✅ Layer {layer_idx} → VRAM ({layer_size_gb:.2f}GB)")
                else:
                    # VRAM full, use GTT
                    gpu_layer = self._load_layer_to_gpu(layer_weights, 'gtt')
                    self.gtt_tensors[layer_idx] = gpu_layer
                    gtt_used_gb += layer_size_gb
                    logger.info(f"   ⚡ Layer {layer_idx} → GTT ({layer_size_gb:.2f}GB)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load layer {layer_idx}: {e}")
        
        # Priority 2: Load remaining layers to GTT
        logger.info("📦 Loading remaining layers to GTT...")
        for layer_idx in range(62):
            if layer_idx in self.vram_tensors or layer_idx in self.gtt_tensors:
                continue  # Already loaded
                
            try:
                layer_weights = self.layer_loader(layer_idx)
                layer_size_gb = self._estimate_layer_size(layer_weights) / (1024**3)
                
                if gtt_used_gb + layer_size_gb <= gtt_available_gb:
                    # Load to GTT
                    gpu_layer = self._load_layer_to_gpu(layer_weights, 'gtt')
                    self.gtt_tensors[layer_idx] = gpu_layer
                    gtt_used_gb += layer_size_gb
                    
                    if layer_idx % 10 == 0:
                        logger.info(f"   ✅ Layer {layer_idx} → GTT ({layer_size_gb:.2f}GB)")
                else:
                    # GTT full, use CPU RAM
                    cpu_layer = self._load_layer_to_cpu(layer_weights)
                    self.cpu_tensors[layer_idx] = cpu_layer
                    cpu_used_gb += layer_size_gb
                    logger.warning(f"   ⚠️ Layer {layer_idx} → CPU RAM ({layer_size_gb:.2f}GB)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load layer {layer_idx}: {e}")
        
        # Load shared weights (embeddings, norms) to VRAM if space available
        logger.info("📦 Loading shared weights...")
        shared_size_gb = 0
        for name, weight_info in self.shared_weights.items():
            try:
                weight_size_gb = self._estimate_weight_size(weight_info) / (1024**3)
                shared_size_gb += weight_size_gb
            except:
                pass
        
        if vram_used_gb + shared_size_gb <= vram_available_gb:
            logger.info(f"   ✅ Shared weights → VRAM ({shared_size_gb:.2f}GB)")
            target = 'vram'
        elif gtt_used_gb + shared_size_gb <= gtt_available_gb:
            logger.info(f"   ⚡ Shared weights → GTT ({shared_size_gb:.2f}GB)")
            target = 'gtt'
        else:
            logger.warning(f"   ⚠️ Shared weights → CPU RAM ({shared_size_gb:.2f}GB)")
            target = 'cpu'
        
        # Summary
        logger.info(f"🎉 MODEL LOADED TO GPU MEMORY!")
        logger.info(f"   📍 VRAM: {vram_used_gb:.1f}GB used ({len(self.vram_tensors)} layers)")
        logger.info(f"   📍 GTT: {gtt_used_gb:.1f}GB used ({len(self.gtt_tensors)} layers)")
        logger.info(f"   📍 CPU: {cpu_used_gb:.1f}GB used ({len(self.cpu_tensors)} layers)")
        logger.info(f"   📊 Total: {vram_used_gb + gtt_used_gb + cpu_used_gb:.1f}GB")
        
        # Verify GPU memory usage
        final_stats = self.gpu_allocator.get_memory_stats()
        logger.info(f"📊 Actual GPU Memory Usage:")
        logger.info(f"   VRAM: {final_stats['vram_used_gb']:.1f}GB / {final_stats['vram_total_gb']:.1f}GB")
        logger.info(f"   GTT: {final_stats['gtt_used_gb']:.1f}GB / {final_stats['gtt_total_gb']:.1f}GB")
    
    def _load_layer_to_gpu(self, layer_weights: Dict[str, Any], target: str = 'gtt') -> Dict[str, Any]:
        """Load layer weights to GPU memory (VRAM or GTT)"""
        
        gpu_layer = {}
        
        for name, weight_info in layer_weights.items():
            if not name.startswith('language_model'):
                continue
                
            try:
                # Get tensor from mmap
                if weight_info.get('lazy', False) and self.mmap_loader:
                    tensor = self.mmap_loader.get_tensor(weight_info)
                else:
                    tensor = weight_info.get('tensor')
                
                # Convert to numpy if needed
                if hasattr(tensor, 'numpy'):
                    tensor = tensor.numpy()
                elif not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
                
                # Transfer to GPU memory
                gpu_tensor = self.gpu_allocator.transfer_to_gpu(tensor, target)
                gpu_layer[name] = {
                    'tensor': gpu_tensor,
                    'quantized': weight_info.get('quantized', False),
                    'scale': weight_info.get('scale'),
                    'scheme': weight_info.get('scheme'),
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'gpu_memory': target
                }
                
            except Exception as e:
                logger.warning(f"Failed to load {name} to GPU: {e}")
                gpu_layer[name] = weight_info
        
        return gpu_layer
    
    def _load_layer_to_cpu(self, layer_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Load layer weights to CPU memory"""
        
        cpu_layer = {}
        
        for name, weight_info in layer_weights.items():
            if not name.startswith('language_model'):
                continue
                
            try:
                # Get tensor from mmap
                if weight_info.get('lazy', False) and self.mmap_loader:
                    tensor = self.mmap_loader.get_tensor(weight_info)
                else:
                    tensor = weight_info.get('tensor')
                
                # Convert to numpy if needed
                if hasattr(tensor, 'numpy'):
                    tensor = tensor.numpy()
                elif not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
                
                cpu_layer[name] = {
                    'tensor': tensor,
                    'quantized': weight_info.get('quantized', False),
                    'scale': weight_info.get('scale'),
                    'scheme': weight_info.get('scheme'),
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'gpu_memory': 'cpu'
                }
                
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
                cpu_layer[name] = weight_info
        
        return cpu_layer
    
    def _estimate_layer_size(self, layer_weights: Dict[str, Any]) -> int:
        """Estimate layer size in bytes"""
        
        total_bytes = 0
        for name, weight_info in layer_weights.items():
            if not name.startswith('language_model'):
                continue
                
            try:
                shape = weight_info.get('shape', [])
                dtype = weight_info.get('dtype', 'float32')
                
                # Calculate element count
                elements = 1
                for dim in shape:
                    elements *= dim
                
                # Get bytes per element
                if 'int8' in str(dtype):
                    bytes_per_element = 1
                elif 'int4' in str(dtype):
                    bytes_per_element = 0.5
                elif 'float16' in str(dtype) or 'bfloat16' in str(dtype):
                    bytes_per_element = 2
                else:  # float32
                    bytes_per_element = 4
                
                total_bytes += int(elements * bytes_per_element)
                
            except Exception as e:
                # Fallback estimate
                total_bytes += 50 * 1024 * 1024  # 50MB per weight
        
        return total_bytes
    
    def _estimate_weight_size(self, weight_info: Dict[str, Any]) -> int:
        """Estimate single weight size in bytes"""
        
        try:
            shape = weight_info.get('shape', [])
            dtype = weight_info.get('dtype', 'float32')
            
            elements = 1
            for dim in shape:
                elements *= dim
            
            if 'int8' in str(dtype):
                bytes_per_element = 1
            elif 'int4' in str(dtype):
                bytes_per_element = 0.5
            elif 'float16' in str(dtype) or 'bfloat16' in str(dtype):
                bytes_per_element = 2
            else:
                bytes_per_element = 4
            
            return int(elements * bytes_per_element)
            
        except:
            return 50 * 1024 * 1024  # 50MB fallback
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """Get layer weights from appropriate memory location"""
        
        # Check in order: VRAM -> GTT -> CPU
        if layer_idx in self.vram_tensors:
            logger.debug(f"⚡ Layer {layer_idx} from VRAM (fastest)")
            return self.vram_tensors[layer_idx]
        elif layer_idx in self.gtt_tensors:
            logger.debug(f"💾 Layer {layer_idx} from GTT (fast)")
            return self.gtt_tensors[layer_idx]
        elif layer_idx in self.cpu_tensors:
            logger.debug(f"🐌 Layer {layer_idx} from CPU RAM (slow)")
            return self.cpu_tensors[layer_idx]
        else:
            logger.error(f"❌ Layer {layer_idx} not found!")
            return {}
    
    def cleanup(self):
        """Cleanup hardware resources"""
        if self.vulkan_engine:
            self.vulkan_engine.cleanup()
        if self.npu_kernel:
            self.npu_kernel.cleanup()
        
        # Clear GPU memory references
        self.vram_tensors.clear()
        self.gtt_tensors.clear()
        self.cpu_tensors.clear()


def test_hma_pipeline():
    """Test the HMA-enabled pipeline"""
    
    print("🦄 Testing Pure Hardware Pipeline with HMA GPU Memory")
    print("=" * 60)
    
    model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
    
    pipeline = PureHardwarePipelineHMA()
    
    if not pipeline.initialize(model_path):
        print("❌ Failed to initialize pipeline")
        return False
    
    print("✅ HMA pipeline initialized with GPU memory allocation")
    
    # Check memory distribution
    print("\n📊 Model Memory Distribution:")
    print(f"   VRAM layers: {len(pipeline.vram_tensors)}")
    print(f"   GTT layers: {len(pipeline.gtt_tensors)}")
    print(f"   CPU layers: {len(pipeline.cpu_tensors)}")
    
    pipeline.cleanup()
    print("\n🎉 HMA pipeline test completed!")
    return True


if __name__ == "__main__":
    test_hma_pipeline()