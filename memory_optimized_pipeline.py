#!/usr/bin/env python3
"""
Memory Optimized Pipeline - Fix memory allocation inefficiencies
Implement proper VRAM/GTT allocation strategy and prevent memory bloat
"""

import numpy as np
import logging
import time
import os
import gc
import mmap
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import our best performing pipeline as base
from advanced_kernel_optimization import AdvancedKernelOptimizedPipeline

logger = logging.getLogger(__name__)

class MemoryOptimizedPipeline(AdvancedKernelOptimizedPipeline):
    """Pipeline with optimized memory allocation strategy"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_strategy = 'vram_priority'
        self.vram_budget_gb = 14.5  # Conservative VRAM budget
        self.gtt_budget_gb = 11.5   # Conservative GTT budget
        self.allocated_vram_gb = 0.0
        self.allocated_gtt_gb = 0.0
        self.memory_mapped_files = []  # Track all mmap files for cleanup
        self.buffer_pool = {}  # Buffer pooling to reduce fragmentation
        
        logger.info("🚀 Memory Optimized Pipeline: Fixing allocation inefficiencies")
    
    def initialize(self, model_path: str) -> bool:
        """Initialize with optimized memory allocation"""
        logger.info("🎯 Starting memory-optimized model loading...")
        
        # Clear any existing cache before starting
        self._clear_system_cache()
        
        # Pre-allocate buffer pools
        self._initialize_buffer_pools()
        
        # Use optimized loading strategy
        success = self._optimized_model_loading(model_path)
        
        if success:
            # Post-loading cleanup
            self._post_loading_cleanup()
            
            # Verify memory allocation
            self._verify_memory_allocation()
        
        return success
    
    def _clear_system_cache(self):
        """Clear system cache before loading"""
        try:
            logger.info("🧹 Clearing system cache before loading...")
            
            # Force garbage collection first
            for _ in range(3):
                gc.collect()
            
            # Clear file cache
            os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
            
            logger.info("   ✅ System cache cleared")
            
        except Exception as e:
            logger.warning(f"Cache clearing: {e}")
    
    def _initialize_buffer_pools(self):
        """Initialize buffer pools to reduce allocation overhead"""
        try:
            logger.info("📦 Initializing optimized buffer pools...")
            
            # Pre-allocate common buffer sizes to reduce fragmentation
            common_sizes = [
                (1024 * 1024, 'small'),      # 1MB buffers
                (16 * 1024 * 1024, 'medium'), # 16MB buffers  
                (256 * 1024 * 1024, 'large'), # 256MB buffers
                (1024 * 1024 * 1024, 'huge')  # 1GB buffers
            ]
            
            for size, name in common_sizes:
                self.buffer_pool[name] = {
                    'size': size,
                    'allocated': False,
                    'pool': []
                }
            
            logger.info("   ✅ Buffer pools initialized")
            
        except Exception as e:
            logger.warning(f"Buffer pool initialization: {e}")
    
    def _optimized_model_loading(self, model_path: str) -> bool:
        """Optimized model loading with proper memory allocation"""
        try:
            logger.info("🚀 Phase 1: Initialize base pipeline components...")
            
            # Initialize Vulkan engine first
            if not self._initialize_vulkan_engine():
                return False
            
            # Initialize NPU components
            if not self._initialize_npu_components():
                return False
            
            logger.info("🚀 Phase 2: Load model with optimized allocation...")
            
            # Use optimized loader with immediate cleanup
            success = self._load_model_with_immediate_cleanup(model_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Optimized model loading failed: {e}")
            return False
    
    def _initialize_vulkan_engine(self) -> bool:
        """Initialize Vulkan engine with memory tracking"""
        try:
            from real_vulkan_matrix_compute import VulkanMatrixCompute
            
            logger.info("   🎮 Initializing Vulkan engine...")
            self.vulkan_engine = VulkanMatrixCompute()
            self.vulkan_engine.use_fp16 = False
            
            # Track VRAM allocation from Vulkan
            if hasattr(self.vulkan_engine, 'get_memory_usage'):
                memory_usage = self.vulkan_engine.get_memory_usage()
                self.allocated_vram_gb += memory_usage.get('vram_gb', 2.3)  # Default buffer pool
            
            logger.info("   ✅ Vulkan engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"Vulkan initialization failed: {e}")
            return False
    
    def _initialize_npu_components(self) -> bool:
        """Initialize NPU components without excessive allocation"""
        try:
            from npu_attention_kernel_optimized import NPUAttentionKernel
            
            logger.info("   🧠 Initializing NPU components...")
            
            # Initialize NPU kernel with minimal allocation
            self.npu_kernel = NPUAttentionKernel(
                seq_length=256,
                model_dim=5376,
                num_heads=32,
                head_dim=168
            )
            
            logger.info("   ✅ NPU components initialized")
            return True
            
        except Exception as e:
            logger.warning(f"NPU initialization: {e}")
            # Continue without NPU if it fails
            self.npu_kernel = None
            return True
    
    def _load_model_with_immediate_cleanup(self, model_path: str) -> bool:
        """Load model with immediate cleanup of memory-mapped files"""
        try:
            from pure_mmap_loader import PureMmapLoader
            
            logger.info("   📂 Loading model with immediate cleanup strategy...")
            
            # Initialize loader
            self.loader = PureMmapLoader()
            self.loader.load_model_structure(model_path)
            
            # Track memory allocation
            initial_vram = self.allocated_vram_gb
            initial_gtt = self.allocated_gtt_gb
            
            # Load with optimized allocation strategy
            success = self._load_weights_with_smart_allocation()
            
            # Immediate cleanup of memory-mapped files
            self._immediate_mmap_cleanup()
            
            if success:
                final_vram = self.allocated_vram_gb
                final_gtt = self.allocated_gtt_gb
                
                logger.info(f"   📊 Memory allocation summary:")
                logger.info(f"      VRAM: {final_vram:.1f} GB (target: {self.vram_budget_gb:.1f} GB)")
                logger.info(f"      GTT:  {final_gtt:.1f} GB (target: {self.gtt_budget_gb:.1f} GB)")
                logger.info(f"      Total: {final_vram + final_gtt:.1f} GB")
            
            return success
            
        except Exception as e:
            logger.error(f"Model loading with cleanup failed: {e}")
            return False
    
    def _load_weights_with_smart_allocation(self) -> bool:
        """Load weights with smart VRAM/GTT allocation strategy"""
        try:
            # Load shared weights first (embeddings, etc.)
            self._load_shared_weights_optimized()
            
            # Load layers with priority allocation
            self._load_layers_with_priority_allocation()
            
            return True
            
        except Exception as e:
            logger.error(f"Smart weight allocation failed: {e}")
            return False
    
    def _load_shared_weights_optimized(self):
        """Load shared weights to VRAM with immediate cleanup"""
        logger.info("   📦 Loading shared weights to VRAM...")
        
        # Load embeddings and shared components
        shared_weights = self.loader.get_shared_weights()
        
        for name, weight_info in shared_weights.items():
            if 'embed_tokens' in name or 'norm' in name:
                # Load to VRAM and track allocation
                tensor_data = self._load_tensor_with_cleanup(weight_info)
                
                if tensor_data is not None:
                    # Allocate to VRAM
                    gpu_buffer = self._allocate_to_vram(tensor_data, name)
                    if gpu_buffer:
                        self.gpu_buffers[name] = gpu_buffer
                        size_gb = tensor_data.nbytes / (1024**3)
                        self.allocated_vram_gb += size_gb
                        logger.info(f"      ✅ {name}: {size_gb:.2f} GB → VRAM")
        
        logger.info(f"   ✅ Shared weights loaded: {self.allocated_vram_gb:.1f} GB VRAM used")
    
    def _load_layers_with_priority_allocation(self):
        """Load layers with VRAM priority strategy"""
        logger.info("   🔄 Loading layers with priority allocation...")
        
        # Priority allocation strategy
        vram_layers = list(range(0, 25))     # First 25 layers to VRAM
        gtt_layers = list(range(25, 62))     # Remaining layers to GTT
        
        # Load VRAM layers first
        logger.info(f"      🔥 Loading layers {vram_layers[0]}-{vram_layers[-1]} to VRAM...")
        for layer_idx in vram_layers:
            if self.allocated_vram_gb >= self.vram_budget_gb:
                logger.info(f"         ⚠️ VRAM budget reached, moving layer {layer_idx} to GTT")
                gtt_layers.insert(0, layer_idx)
                continue
            
            success = self._load_layer_to_vram(layer_idx)
            if not success:
                logger.warning(f"         ⚠️ Layer {layer_idx} failed VRAM allocation, trying GTT")
                gtt_layers.insert(0, layer_idx)
        
        # Load GTT layers
        logger.info(f"      ❄️ Loading layers {gtt_layers[0]}-{gtt_layers[-1]} to GTT...")
        for layer_idx in gtt_layers:
            if self.allocated_gtt_gb >= self.gtt_budget_gb:
                logger.error(f"         ❌ GTT budget exceeded at layer {layer_idx}")
                return False
            
            success = self._load_layer_to_gtt(layer_idx)
            if not success:
                logger.error(f"         ❌ Layer {layer_idx} failed GTT allocation")
                return False
        
        logger.info(f"   ✅ All layers loaded: {self.allocated_vram_gb:.1f} GB VRAM, {self.allocated_gtt_gb:.1f} GB GTT")
        return True
    
    def _load_tensor_with_cleanup(self, weight_info) -> Optional[np.ndarray]:
        """Load tensor data and immediately cleanup memory-mapped file"""
        try:
            # Get tensor data
            tensor_data = self.loader.get_tensor(weight_info)
            
            # Make a copy to avoid holding reference to mmap
            tensor_copy = np.copy(tensor_data)
            
            # Force cleanup of original tensor
            del tensor_data
            
            return tensor_copy
            
        except Exception as e:
            logger.warning(f"Tensor loading with cleanup failed: {e}")
            return None
    
    def _allocate_to_vram(self, tensor_data: np.ndarray, name: str) -> Optional[Tuple]:
        """Allocate tensor to VRAM with tracking"""
        try:
            if hasattr(self.vulkan_engine, '_allocate_gpu_memory'):
                gpu_buffer = self.vulkan_engine._allocate_gpu_memory(tensor_data)
                return gpu_buffer
            else:
                logger.warning(f"VRAM allocation not available for {name}")
                return None
                
        except Exception as e:
            logger.warning(f"VRAM allocation failed for {name}: {e}")
            return None
    
    def _allocate_to_gtt(self, tensor_data: np.ndarray, name: str) -> Optional[Tuple]:
        """Allocate tensor to GTT with tracking"""
        try:
            if hasattr(self.vulkan_engine, '_allocate_gtt_memory'):
                gtt_buffer = self.vulkan_engine._allocate_gtt_memory(tensor_data)
                return gtt_buffer
            else:
                logger.warning(f"GTT allocation not available for {name}")
                return None
                
        except Exception as e:
            logger.warning(f"GTT allocation failed for {name}: {e}")
            return None
    
    def _load_layer_to_vram(self, layer_idx: int) -> bool:
        """Load a single layer to VRAM"""
        try:
            layer_weights = self.loader.get_layer_weights(layer_idx)
            layer_size_gb = 0.0
            
            for name, weight_info in layer_weights.items():
                tensor_data = self._load_tensor_with_cleanup(weight_info)
                if tensor_data is not None:
                    gpu_buffer = self._allocate_to_vram(tensor_data, name)
                    if gpu_buffer:
                        buffer_key = f'layer_{layer_idx}_{name}'
                        self.gpu_buffers[buffer_key] = gpu_buffer
                        size_gb = tensor_data.nbytes / (1024**3)
                        layer_size_gb += size_gb
            
            self.allocated_vram_gb += layer_size_gb
            self.layer_weights_gpu[layer_idx] = True
            
            logger.debug(f"         ✅ Layer {layer_idx}: {layer_size_gb:.2f} GB → VRAM")
            return True
            
        except Exception as e:
            logger.warning(f"Layer {layer_idx} VRAM loading failed: {e}")
            return False
    
    def _load_layer_to_gtt(self, layer_idx: int) -> bool:
        """Load a single layer to GTT"""
        try:
            layer_weights = self.loader.get_layer_weights(layer_idx)
            layer_size_gb = 0.0
            
            for name, weight_info in layer_weights.items():
                tensor_data = self._load_tensor_with_cleanup(weight_info)
                if tensor_data is not None:
                    gtt_buffer = self._allocate_to_gtt(tensor_data, name)
                    if gtt_buffer:
                        buffer_key = f'layer_{layer_idx}_{name}'
                        self.gpu_buffers[buffer_key] = gtt_buffer
                        size_gb = tensor_data.nbytes / (1024**3)
                        layer_size_gb += size_gb
            
            self.allocated_gtt_gb += layer_size_gb
            self.layer_weights_gpu[layer_idx] = True
            
            logger.debug(f"         ✅ Layer {layer_idx}: {layer_size_gb:.2f} GB → GTT")
            return True
            
        except Exception as e:
            logger.warning(f"Layer {layer_idx} GTT loading failed: {e}")
            return False
    
    def _immediate_mmap_cleanup(self):
        """Immediately cleanup all memory-mapped files"""
        try:
            logger.info("   🧹 Immediate memory-mapped file cleanup...")
            
            # Close all tracked mmap files
            for mmap_file in self.memory_mapped_files:
                try:
                    if hasattr(mmap_file, 'close'):
                        mmap_file.close()
                except:
                    pass
            
            self.memory_mapped_files.clear()
            
            # Force garbage collection
            for _ in range(3):
                gc.collect()
            
            # Close any remaining file handles in loader
            if hasattr(self.loader, 'close_all_files'):
                self.loader.close_all_files()
            elif hasattr(self.loader, 'memory_maps'):
                for mmap_obj in self.loader.memory_maps.values():
                    try:
                        if hasattr(mmap_obj, 'close'):
                            mmap_obj.close()
                    except:
                        pass
                self.loader.memory_maps.clear()
            
            logger.info("      ✅ Memory-mapped files cleaned up")
            
        except Exception as e:
            logger.warning(f"Memory-mapped cleanup: {e}")
    
    def _post_loading_cleanup(self):
        """Post-loading cleanup and optimization"""
        try:
            logger.info("🧹 Post-loading cleanup and optimization...")
            
            # Final memory cleanup
            self._final_memory_cleanup()
            
            # Initialize remaining components
            self._initialize_remaining_components()
            
            logger.info("   ✅ Post-loading cleanup complete")
            
        except Exception as e:
            logger.warning(f"Post-loading cleanup: {e}")
    
    def _final_memory_cleanup(self):
        """Final memory cleanup to release cache"""
        try:
            # Force aggressive garbage collection
            for _ in range(5):
                gc.collect()
            
            # Clear file cache again
            os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
            
            # Clear any Python caches
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            logger.info("      ✅ Final memory cleanup complete")
            
        except Exception as e:
            logger.warning(f"Final memory cleanup: {e}")
    
    def _initialize_remaining_components(self):
        """Initialize remaining pipeline components"""
        try:
            # Initialize layer fusion if available
            if hasattr(self, '_initialize_layer_fusion'):
                self._initialize_layer_fusion()
            
            # Initialize memory optimizations
            if hasattr(self, '_optimize_memory_patterns'):
                self._optimize_memory_patterns()
            
        except Exception as e:
            logger.warning(f"Remaining component initialization: {e}")
    
    def _verify_memory_allocation(self):
        """Verify memory allocation is within budget"""
        logger.info("📊 Memory allocation verification...")
        
        # Check VRAM allocation
        if self.allocated_vram_gb > self.vram_budget_gb:
            logger.warning(f"   ⚠️ VRAM over budget: {self.allocated_vram_gb:.1f} GB > {self.vram_budget_gb:.1f} GB")
        else:
            logger.info(f"   ✅ VRAM within budget: {self.allocated_vram_gb:.1f} GB / {self.vram_budget_gb:.1f} GB")
        
        # Check GTT allocation  
        if self.allocated_gtt_gb > self.gtt_budget_gb:
            logger.warning(f"   ⚠️ GTT over budget: {self.allocated_gtt_gb:.1f} GB > {self.gtt_budget_gb:.1f} GB")
        else:
            logger.info(f"   ✅ GTT within budget: {self.allocated_gtt_gb:.1f} GB / {self.gtt_budget_gb:.1f} GB")
        
        total_allocated = self.allocated_vram_gb + self.allocated_gtt_gb
        logger.info(f"   📊 Total allocated: {total_allocated:.1f} GB")
        
        # Verify system memory usage
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"   💾 System RAM: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent:.1f}%)")
            
            # Check file cache
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('Cached:'):
                    cache_gb = int(line.split()[1]) / (1024**2)
                    logger.info(f"   📁 File cache: {cache_gb:.1f} GB")
                    if cache_gb > 5.0:
                        logger.warning(f"      ⚠️ File cache still elevated: {cache_gb:.1f} GB")
                    break
            
        except Exception as e:
            logger.warning(f"System memory verification: {e}")
    
    def cleanup(self):
        """Clean up all memory allocations"""
        logger.info("🧹 Cleaning up memory allocations...")
        
        try:
            # Cleanup memory-mapped files
            self._immediate_mmap_cleanup()
            
            # Clear GPU buffers
            if hasattr(self, 'gpu_buffers'):
                self.gpu_buffers.clear()
            
            # Reset allocation tracking
            self.allocated_vram_gb = 0.0
            self.allocated_gtt_gb = 0.0
            
            # Final cache clear
            os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
            
            # Call parent cleanup
            super().cleanup()
            
            logger.info("   ✅ Memory cleanup complete")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


def test_memory_optimized_pipeline():
    """Test memory optimized pipeline"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Testing Memory Optimized Pipeline")
    
    # Initialize with memory optimization
    pipeline = MemoryOptimizedPipeline(enable_parallelism=True, cache_size=8)
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model with memory optimization...")
    start = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    load_time = time.time() - start
    logger.info(f"✅ Model loaded in {load_time:.1f}s with optimized memory allocation")
    
    # Run performance test
    logger.info("🔥 Testing performance with optimized memory...")
    test_input = np.random.randn(1, 1, 5376).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        output, _ = pipeline.forward_layer_advanced_optimized(0, test_input)
    
    # Benchmark
    times = []
    for _ in range(30):
        start = time.perf_counter()
        output, _ = pipeline.forward_layer_advanced_optimized(0, test_input)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    tps = 1.0 / (avg_time * 62)
    
    logger.info(f"📊 Memory Optimized Results:")
    logger.info(f"   Layer time: {avg_time*1000:.2f}ms")
    logger.info(f"   Estimated TPS: {tps:.1f}")
    logger.info(f"   Memory efficiency: Optimized VRAM/GTT allocation")
    
    # Verify final memory state
    pipeline._verify_memory_allocation()
    
    # Cleanup
    pipeline.cleanup()
    
    return tps


if __name__ == "__main__":
    test_memory_optimized_pipeline()