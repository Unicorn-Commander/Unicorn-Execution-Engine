#!/usr/bin/env python3
"""
Pure Hardware Pipeline with DIRECT GPU Loading
- Maps safetensor files directly to GPU memory
- No CPU intermediate copies
- Parallel loading with multiple threads
- Proper memory usage (26GB total)
"""

import numpy as np
import logging
import time
import os
import mmap
import struct
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from pure_mmap_loader import PureMemoryMappedLoader
from real_vulkan_matrix_compute import VulkanMatrixCompute
from npu_attention_kernel_real import NPUAttentionKernelReal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PureHardwarePipelineDirectGPU:
    def __init__(self):
        """Initialize the pure hardware pipeline with direct GPU loading"""
        logger.info("🚀 Initializing Direct GPU Loading Pipeline")
        
        # Core components
        self.vulkan_engine = None
        self.npu_kernel = None
        self.loader = None
        
        # Model structure
        self.config = None
        self.tokenizer = None
        self.layer_weights_gpu = {}
        self.shared_weights_gpu = {}
        self.gpu_buffers = {}  # Stores GPU buffer handles
        
        # Initialize hardware
        self._initialize_hardware()
        
    def _initialize_hardware(self):
        """Initialize GPU and NPU hardware"""
        # Initialize Vulkan GPU engine
        try:
            self.vulkan_engine = VulkanMatrixCompute()
            if self.vulkan_engine.initialize():
                logger.info("✅ Vulkan iGPU engine initialized")
            else:
                raise RuntimeError("Failed to initialize Vulkan")
        except Exception as e:
            logger.error(f"❌ Vulkan initialization failed: {e}")
            raise
            
        # Initialize NPU kernel - Try real hardware first
        try:
            self.npu_kernel = NPUAttentionKernelReal()
            if self.npu_kernel.initialize():
                logger.info("✅ Real NPU kernel initialized")
            else:
                logger.warning("⚠️ NPU initialization failed, will use GPU fallback")
                self.npu_kernel = None
        except Exception as e:
            logger.warning(f"⚠️ NPU not available: {e}, using GPU fallback")
            self.npu_kernel = None

    def initialize(self, model_path: str) -> bool:
        """Initialize model with direct GPU loading"""
        try:
            # Initialize loader
            self.loader = PureMemoryMappedLoader(model_path)
            
            # Load config
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    logger.info(f"✅ Loaded config: {self.config.get('num_hidden_layers', 'unknown')} layers")
            
            # Load model weights directly to GPU
            logger.info("🔄 Loading model directly to GPU memory...")
            start_time = time.time()
            
            # Get baseline GPU memory
            baseline_vram = self._get_gpu_memory_usage()
            logger.info(f"📊 Baseline GPU memory: VRAM {baseline_vram:.1f}MB")
            
            # Load weights with parallel threads
            self._load_all_weights_parallel()
            
            # Check final memory usage
            final_vram = self._get_gpu_memory_usage()
            vram_increase = final_vram - baseline_vram
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Model loaded in {elapsed:.1f}s")
            logger.info(f"📊 GPU memory increase: {vram_increase/1024:.1f}GB")
            logger.info(f"   Expected: ~26GB, Actual: {vram_increase/1024:.1f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_all_weights_parallel(self):
        """Load all weights to GPU using parallel threads"""
        # Get all tensor infos
        all_tensors = []
        
        # Shared weights (embeddings, etc)
        for name, info in self.loader.tensors.items():
            if 'layer' not in name or 'embed_tokens' in name:
                all_tensors.append((name, info, True))  # Use VRAM for shared
        
        # Layer weights
        num_layers = self.config.get('num_hidden_layers', 62)
        for layer_idx in range(num_layers):
            layer_tensors = self.loader._get_layer_tensors(layer_idx)
            for name, info in layer_tensors.items():
                # First 20 layers to VRAM, rest to GTT
                use_vram = layer_idx < 20
                all_tensors.append((name, info, use_vram))
        
        # Load in parallel with progress tracking
        total_size = 0
        loaded = 0
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all loading tasks
            futures = {}
            for name, info, use_vram in all_tensors:
                future = executor.submit(self._load_tensor_direct_gpu, name, info, use_vram)
                futures[future] = (name, info)
            
            # Process completed tasks
            for future in as_completed(futures):
                name, info = futures[future]
                try:
                    size_mb = future.result()
                    if size_mb > 0:
                        total_size += size_mb
                        loaded += 1
                        if loaded % 100 == 0:
                            logger.info(f"   Loaded {loaded}/{len(all_tensors)} tensors, {total_size/1024:.1f}GB")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
        
        logger.info(f"✅ Loaded {loaded} tensors, total {total_size/1024:.1f}GB")

    def _load_tensor_direct_gpu(self, name: str, tensor_info: dict, use_vram: bool) -> float:
        """Load a single tensor directly to GPU memory"""
        try:
            # Get tensor metadata
            shape = tensor_info.get('shape', [])
            dtype = tensor_info.get('dtype', 'float32')
            data_offsets = tensor_info.get('data_offsets', [0, 0])
            filename = tensor_info.get('filename', '')
            
            # Calculate size
            elements = 1
            for dim in shape:
                elements *= dim
            
            dtype_sizes = {
                'float32': 4, 'float16': 2, 'bfloat16': 2,
                'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1,
                'F32': 4, 'F16': 2, 'BF16': 2, 'I8': 1, 'U8': 1
            }
            bytes_per_element = dtype_sizes.get(dtype, 4)
            tensor_size = elements * bytes_per_element
            size_mb = tensor_size / (1024 * 1024)
            
            # Get the memory-mapped file
            if filename not in self.loader.mapped_files:
                logger.warning(f"File {filename} not mapped")
                return 0
            
            mapped_file = self.loader.mapped_files[filename]
            
            # Create a view of the tensor data without copying
            start_offset = data_offsets[0]
            end_offset = data_offsets[1]
            
            # Create numpy array that views the mmap data directly
            if dtype == 'I8' or dtype == 'int8':
                np_dtype = np.int8
            elif dtype == 'U8' or dtype == 'uint8':
                np_dtype = np.uint8
            elif dtype == 'F16' or dtype == 'float16':
                np_dtype = np.float16
            elif dtype == 'F32' or dtype == 'float32':
                np_dtype = np.float32
            else:
                np_dtype = np.float32
            
            # Create array from mmap buffer without copy
            tensor_view = np.frombuffer(
                mapped_file, 
                dtype=np_dtype,
                count=elements,
                offset=start_offset
            ).reshape(shape)
            
            # Allocate GPU memory and copy directly
            if use_vram:
                gpu_buffer_info = self.vulkan_engine._allocate_gpu_memory(tensor_view)
            else:
                gpu_buffer_info = self.vulkan_engine._allocate_gtt_memory(tensor_view)
            
            # Store buffer info
            self.gpu_buffers[name] = {
                'buffer_info': gpu_buffer_info,
                'shape': shape,
                'dtype': dtype,
                'size_mb': size_mb
            }
            
            # Organize by layer
            if 'layer' in name and 'layers.' in name:
                # Extract layer index
                layer_idx = int(name.split('layers.')[1].split('.')[0])
                if layer_idx not in self.layer_weights_gpu:
                    self.layer_weights_gpu[layer_idx] = {}
                self.layer_weights_gpu[layer_idx][name] = name
            else:
                self.shared_weights_gpu[name] = name
            
            return size_mb
            
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            return 0

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            # Check Vulkan reported usage
            if hasattr(self.vulkan_engine, 'memory_usage_mb'):
                return self.vulkan_engine.memory_usage_mb
            
            # Fallback to system query
            result = os.popen("cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null").read().strip()
            if result:
                return int(result) / (1024 * 1024)  # Convert bytes to MB
            
            return 0
        except:
            return 0

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up pipeline resources...")
        
        if self.vulkan_engine:
            self.vulkan_engine.cleanup()
        
        if self.npu_kernel:
            self.npu_kernel.cleanup()
        
        if self.loader:
            self.loader.cleanup()
        
        self.gpu_buffers.clear()
        self.layer_weights_gpu.clear()
        self.shared_weights_gpu.clear()
        
        logger.info("✅ Cleanup complete")


def main():
    """Test direct GPU loading"""
    pipeline = PureHardwarePipelineDirectGPU()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if pipeline.initialize(model_path):
        logger.info("✅ Pipeline initialized successfully!")
        
        # Quick memory check
        import subprocess
        try:
            # Check VRAM usage
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True, timeout=2)
            output = result.stdout
            for line in output.split('\n'):
                if 'vram' in line.lower() or 'gtt' in line.lower():
                    logger.info(f"   {line.strip()}")
        except:
            pass
        
        # Keep alive for a moment to check memory
        time.sleep(5)
    else:
        logger.error("❌ Pipeline initialization failed")
    
    pipeline.cleanup()


if __name__ == "__main__":
    main()