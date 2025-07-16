#!/usr/bin/env python3
"""
Simple Fast Loader - No HIP/ROCm complications
Just load the entire model into memory as fast as possible
"""

import os
import torch
import time
import logging
from pathlib import Path
from typing import Dict, Any
from safetensors import safe_open
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use ALL CPU threads
torch.set_num_threads(cpu_count())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFastLoader:
    """Simple fast loader - just get everything into memory ASAP"""
    
    def __init__(self, model_path: str = "./quantized_models/gemma-3-27b-it-layer-by-layer"):
        self.model_path = Path(model_path)
        logger.info(f"🚀 Simple Fast Loader - Target: <15 seconds")
        
    def load_fast(self) -> Dict[str, Any]:
        """Load everything into memory as fast as possible"""
        logger.info("⚡ SIMPLE FAST LOADING - No device complications!")
        
        start_time = time.time()
        
        # Get all files
        all_files = list(self.model_path.glob("*.safetensors"))
        logger.info(f"📂 Found {len(all_files)} files")
        
        all_weights = {}
        total_size_gb = 0
        
        # Use maximum parallelism
        max_workers = min(cpu_count(), len(all_files))
        logger.info(f"🚀 Using {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._load_file_fast, file_path): file_path 
                for file_path in all_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_weights, file_size = future.result()
                    all_weights.update(file_weights)
                    total_size_gb += file_size
                    logger.info(f"✅ {file_path.name}: {len(file_weights)} tensors")
                except Exception as e:
                    logger.error(f"❌ Failed {file_path.name}: {e}")
        
        # Create instant layer accessor
        def instant_layer_access(layer_num: int) -> Dict[str, torch.Tensor]:
            layer_prefix = f"language_model.model.layers.{layer_num}."
            layer_tensors = {}
            for name, weight in all_weights.items():
                if name.startswith(layer_prefix):
                    layer_tensors[name] = {'tensor': weight}
            return layer_tensors
        
        # Find layer count
        layer_numbers = set()
        for weight_name in all_weights.keys():
            if 'language_model.model.layers.' in weight_name:
                try:
                    layer_num = int(weight_name.split('.layers.')[1].split('.')[0])
                    layer_numbers.add(layer_num)
                except:
                    pass
        
        max_layer = max(layer_numbers) if layer_numbers else 0
        load_time = time.time() - start_time
        
        logger.info(f"⚡ SIMPLE LOAD COMPLETE in {load_time:.1f}s")
        logger.info(f"📊 {len(all_weights)} tensors, {total_size_gb:.1f}GB")
        logger.info(f"🚀 Speed: {total_size_gb/load_time:.1f} GB/s")
        
        # Separate shared weights
        shared_weights = {k: {'tensor': v} for k, v in all_weights.items() if 'layers.' not in k}
        
        return {
            'shared_weights': shared_weights,
            'all_weights': {k: {'tensor': v} for k, v in all_weights.items()},
            'layer_count': max_layer + 1,
            'layer_loader': instant_layer_access,
            'hardware_status': {
                'model_size_gb': total_size_gb,
                'load_time_s': load_time,
                'loading_speed_gbps': total_size_gb/load_time,
                'memory_usage_percent': total_size_gb/96*100,
                'quantized_tensors': len(all_weights),
                'dequantized_tensors': 0,
                'mixed_precision': False,
                'cpu_cores_used': max_workers
            }
        }
    
    def _load_file_fast(self, file_path: Path) -> tuple:
        """Load a single file as fast as possible"""
        file_weights = {}
        file_size_gb = file_path.stat().st_size / (1024**3)
        
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not key.endswith('_scale'):  # Skip scale tensors for now
                        tensor = f.get_tensor(key)
                        # Just load into regular memory - keep it simple
                        file_weights[key] = tensor
                        
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}, 0.0
        
        return file_weights, file_size_gb

if __name__ == "__main__":
    loader = SimpleFastLoader()
    model_info = loader.load_fast()
    logger.info("🎉 Simple fast loading test complete!")