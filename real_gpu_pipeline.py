#!/usr/bin/env python3
"""Fixed pipeline that actually loads model weights to GPU memory"""

import logging
import time
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealGPUPipeline(PureHardwarePipelineFixed):
    """Pipeline that forces actual GPU loading, not just references"""
    
    def _load_model_to_gpu(self):
        """Override to ensure actual GPU loading happens"""
        vram_used_mb = 0
        gtt_used_mb = 0
        vram_limit_mb = 16 * 1024  # 16GB
        gtt_limit_mb = 10 * 1024   # 10GB
        
        logger.info("🚀 Starting REAL GPU loading (not just references)...")
        
        # Get the model path and load safetensors directly
        import os
        from safetensors import safe_open
        
        model_files = []
        model_dir = self.loader.model_path if hasattr(self.loader, 'model_path') else "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
        
        # Find all safetensor files
        for f in os.listdir(model_dir):
            if f.endswith('.safetensors'):
                model_files.append(os.path.join(model_dir, f))
        
        logger.info(f"Found {len(model_files)} model files to load")
        
        # Load each file and transfer to GPU
        for file_idx, file_path in enumerate(sorted(model_files)):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    # Skip vision components
                    if 'vision' in tensor_name:
                        continue
                    
                    # Get tensor info
                    tensor = f.get_tensor(tensor_name)
                    size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                    
                    # Decide where to put it
                    if vram_used_mb + size_mb < vram_limit_mb:
                        target = "VRAM"
                        use_vram = True
                        vram_used_mb += size_mb
                    elif gtt_used_mb + size_mb < gtt_limit_mb:
                        target = "GTT"
                        use_vram = False
                        gtt_used_mb += size_mb
                    else:
                        logger.warning(f"No GPU memory left for {tensor_name}")
                        continue
                    
                    # Actually allocate GPU memory and copy data
                    try:
                        # Convert to numpy for Vulkan
                        tensor_np = tensor.numpy().astype(np.float32)
                        
                        # Allocate GPU buffer
                        if use_vram:
                            buffer_info = self.vulkan_engine._allocate_gpu_memory(tensor_np)
                        else:
                            buffer_info = self.vulkan_engine._allocate_gtt_memory(tensor_np)
                        
                        # Store buffer reference
                        self.gpu_buffers[tensor_name] = buffer_info
                        
                        if file_idx == 0 and len(self.gpu_buffers) % 10 == 0:
                            logger.info(f"✅ Loaded {len(self.gpu_buffers)} tensors: VRAM={vram_used_mb/1024:.1f}GB, GTT={gtt_used_mb/1024:.1f}GB")
                    
                    except Exception as e:
                        logger.error(f"Failed to load {tensor_name}: {e}")
        
        logger.info(f"📊 GPU Loading Complete:")
        logger.info(f"   VRAM: {vram_used_mb/1024:.1f}GB / {vram_limit_mb/1024:.1f}GB")
        logger.info(f"   GTT: {gtt_used_mb/1024:.1f}GB / {gtt_limit_mb/1024:.1f}GB")
        logger.info(f"   Total tensors loaded: {len(self.gpu_buffers)}")
        
        # Create layer mapping
        self.layer_weights_gpu = {}
        for i in range(62):
            layer_weights = {}
            for key, buffer in self.gpu_buffers.items():
                if f'layers.{i}.' in key:
                    layer_weights[key] = key
            if layer_weights:
                self.layer_weights_gpu[i] = layer_weights

if __name__ == "__main__":
    # Test the real GPU loading
    pipeline = RealGPUPipeline()
    success = pipeline.initialize(
        model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    )
    
    if success:
        logger.info("✅ Real GPU loading successful!")
    else:
        logger.error("❌ GPU loading failed")
