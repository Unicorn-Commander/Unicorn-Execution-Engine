#!/usr/bin/env python3
"""
Fast Full Model Loader - Load entire 26GB model into GTT/VRAM memory
Uses your 40GB GTT + 16GB VRAM + 96GB RAM for maximum speed
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from safetensors import safe_open
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# Hardware optimization
os.environ['OMP_NUM_THREADS'] = str(cpu_count())
os.environ['MKL_NUM_THREADS'] = str(cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count())
torch.set_num_threads(cpu_count())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastFullModelLoader:
    """Load entire Gemma 3 27B model into memory for instant access"""
    
    def __init__(self, quantized_model_path: str = "./quantized_models/gemma-3-27b-it-layer-by-layer"):
        self.quantized_path = Path(quantized_model_path)
        self.device_assignments = {}
        
        # Hardware detection
        self.npu_available = self._detect_npu()
        self.igpu_available = self._detect_igpu()
        
        logger.info(f"🚀 Fast Full Model Loader initialized")
        logger.info(f"📁 Model path: {self.quantized_path}")
        logger.info(f"⚡ NPU available: {self.npu_available}")
        logger.info(f"🎮 iGPU available: {self.igpu_available}")
        logger.info(f"💾 Shared memory architecture: 96GB total (16GB VRAM + 40GB GTT + 40GB RAM)")
        
    def _detect_npu(self) -> bool:
        """Detect NPU Phoenix hardware"""
        try:
            import subprocess
            result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True)
            return 'Phoenix' in result.stdout and result.returncode == 0
        except:
            return False
    
    def _detect_igpu(self) -> bool:
        """Detect AMD Radeon 780M iGPU"""
        try:
            import subprocess
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            return 'AMD Radeon Graphics' in result.stdout and result.returncode == 0
        except:
            return False
    
    def _get_device_assignment(self, layer_name: str, tensor_name: str) -> str:
        """Determine optimal device for tensor"""
        # NPU: Attention operations
        if any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
            return 'npu'
        # iGPU: FFN operations  
        elif any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
            return 'igpu'
        # CPU: Everything else
        else:
            return 'cpu'
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor, scale: torch.Tensor, scheme: str) -> torch.Tensor:
        """Fast dequantization"""
        if scheme == 'int8_symmetric':
            return quantized_tensor.float() * scale
        elif scheme == 'int4_grouped':
            group_size = 128
            tensor_flat = quantized_tensor.flatten().float()
            dequantized_groups = []
            for i in range(0, len(tensor_flat), group_size):
                group = tensor_flat[i:i+group_size]
                group_scale = scale[i // group_size] if i // group_size < len(scale) else scale[-1]
                dequantized_group = group * group_scale
                dequantized_groups.append(dequantized_group)
            return torch.cat(dequantized_groups).reshape(quantized_tensor.shape)
        elif scheme == 'int8_asymmetric':
            scale_val = scale[0]
            zero_point = scale[1]
            return (quantized_tensor.float() - zero_point) * scale_val
        else:
            return quantized_tensor.float()
    
    def _load_entire_file(self, file_path: Path) -> Tuple[Dict[str, Any], int, float]:
        """Load entire safetensors file with optimal device placement"""
        file_weights = {}
        tensor_count = 0
        total_size = 0
        
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                tensor_names = [key for key in f.keys() if not key.endswith('_scale')]
                metadata = f.metadata()
                
                for tensor_name in tensor_names:
                    try:
                        # Load quantized tensor and scale
                        quantized_tensor = f.get_tensor(tensor_name)
                        scale_name = f"{tensor_name}_scale"
                        scale = f.get_tensor(scale_name) if scale_name in f.keys() else None
                        scheme = metadata.get(tensor_name, 'unknown')
                        
                        # Dequantize
                        if scale is not None:
                            dequantized_tensor = self._dequantize_tensor(quantized_tensor, scale, scheme)
                        else:
                            dequantized_tensor = quantized_tensor.float()
                        
                        # Determine optimal device
                        device = self._get_device_assignment(tensor_name, tensor_name)
                        
                        # Move to optimal memory immediately
                        if device == 'igpu' and torch.cuda.is_available():
                            # FFN weights → VRAM (16GB available)
                            dequantized_tensor = dequantized_tensor.cuda()
                        elif device == 'npu':
                            # NPU weights → Pinned memory for fast transfer
                            dequantized_tensor = dequantized_tensor.pin_memory()
                        else:
                            # CPU weights → GTT memory
                            pass
                        
                        file_weights[tensor_name] = {
                            'tensor': dequantized_tensor,
                            'device': device,
                            'scheme': scheme,
                            'original_shape': dequantized_tensor.shape
                        }
                        
                        # Track memory usage
                        tensor_size = dequantized_tensor.numel() * dequantized_tensor.element_size()
                        total_size += tensor_size
                        tensor_count += 1
                        
                        self.device_assignments[tensor_name] = device
                        
                    except Exception as e:
                        logger.error(f"Failed to load {tensor_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open {file_path}: {e}")
            return {}, 0, 0.0
        
        return file_weights, tensor_count, total_size / (1024**3)  # Convert to GB
    
    def load_full_model_to_memory(self) -> Dict[str, Any]:
        """Load ENTIRE 26GB model into GTT/VRAM memory"""
        logger.info("🚀 Loading ENTIRE Gemma 3 27B model into GTT/VRAM memory")
        logger.info("💾 Target: 96GB shared memory pool (dynamically allocated)")
        
        start_time = time.time()
        
        # Get all model files
        all_files = list(self.quantized_path.glob("*.safetensors"))
        logger.info(f"📂 Found {len(all_files)} safetensors files")
        
        # Load ALL weights in parallel
        all_weights = {}
        total_tensors = 0
        total_size_gb = 0
        
        logger.info("⚡ Loading ALL model weights with 14 parallel threads...")
        
        with ThreadPoolExecutor(max_workers=14) as executor:
            future_to_file = {executor.submit(self._load_entire_file, file_path): file_path for file_path in all_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_weights, file_tensor_count, file_size_gb = future.result()
                    all_weights.update(file_weights)
                    total_tensors += file_tensor_count
                    total_size_gb += file_size_gb
                    logger.info(f"✅ {file_path.name}: {file_tensor_count} tensors, {file_size_gb:.1f}GB")
                except Exception as e:
                    logger.error(f"❌ Failed to load {file_path.name}: {e}")
        
        # Separate shared weights and layers
        shared_weights = {k: v for k, v in all_weights.items() if 'layers.' not in k}
        layer_weights = {k: v for k, v in all_weights.items() if 'layers.' in k}
        
        # Determine layer count
        layer_numbers = set()
        for weight_name in layer_weights.keys():
            if 'language_model.model.layers.' in weight_name:
                try:
                    layer_num = int(weight_name.split('.layers.')[1].split('.')[0])
                    layer_numbers.add(layer_num)
                except:
                    pass
        
        max_layer = max(layer_numbers) if layer_numbers else 0
        
        load_time = time.time() - start_time
        
        logger.info(f"🎉 ENTIRE MODEL LOADED TO MEMORY in {load_time:.1f}s")
        logger.info(f"📊 Total: {total_tensors} tensors, {total_size_gb:.1f}GB loaded")
        logger.info(f"⚡ Speed: {total_size_gb/load_time:.1f} GB/s loading rate")
        logger.info(f"🧠 Memory usage: {total_size_gb:.1f}GB of 96GB available ({total_size_gb/96*100:.1f}%)")
        
        # Create instant layer loader
        def instant_layer_loader(layer_num: int) -> Dict[str, torch.Tensor]:
            """Instant layer access from pre-loaded memory"""
            layer_prefix = f"language_model.model.layers.{layer_num}."
            layer_tensors = {k: v for k, v in all_weights.items() if k.startswith(layer_prefix)}
            logger.info(f"⚡ Instant access to layer {layer_num}: {len(layer_tensors)} tensors")
            return layer_tensors
        
        return {
            'shared_weights': shared_weights,
            'all_weights': all_weights,  # Full model in memory
            'layer_count': max_layer + 1,
            'layer_loader': instant_layer_loader,  # Instant access
            'device_assignments': self.device_assignments,
            'hardware_status': {
                'npu_available': self.npu_available,
                'igpu_available': self.igpu_available,
                'model_size_gb': total_size_gb,
                'load_time_s': load_time,
                'loading_speed_gbps': total_size_gb/load_time,
                'memory_usage_percent': total_size_gb/96*100
            }
        }

def test_fast_loader():
    """Test the fast full model loader"""
    logger.info("🧪 Testing Fast Full Model Loader")
    
    loader = FastFullModelLoader()
    
    # Load entire model
    model_info = loader.load_full_model_to_memory()
    
    logger.info("📊 Model loading complete!")
    logger.info(f"   Shared weights: {len(model_info['shared_weights'])}")
    logger.info(f"   Layer count: {model_info['layer_count']}")
    logger.info(f"   Total size: {model_info['hardware_status']['model_size_gb']:.1f}GB")
    logger.info(f"   Load time: {model_info['hardware_status']['load_time_s']:.1f}s")
    logger.info(f"   Speed: {model_info['hardware_status']['loading_speed_gbps']:.1f} GB/s")
    
    # Test instant layer access
    layer_0 = model_info['layer_loader'](0)
    logger.info(f"✅ Instant layer 0 access: {len(layer_0)} tensors")
    
    return model_info

if __name__ == "__main__":
    test_fast_loader()