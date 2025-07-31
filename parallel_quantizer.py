#!/usr/bin/env python3
"""
Parallel CPU Quantizer for Gemma-3-4B
Uses multiprocessing for maximum CPU utilization
"""

import torch
import time
import logging
import multiprocessing as mp
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Maximize CPU usage
cpu_count = mp.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def should_quantize_int4(tensor_name: str, tensor_size_mb: float) -> bool:
    """FFN weights get INT4 for iGPU"""
    return any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj']) and tensor_size_mb > 1.0

def should_quantize_int8(tensor_name: str, tensor_size_mb: float) -> bool:
    """Attention weights get INT8 for NPU"""
    if any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        return True
    if 'embed_tokens' in tensor_name and tensor_size_mb > 10:
        return True
    return False

def fast_quantize_int8(tensor_data):
    """Fast INT8 quantization using numpy"""
    tensor = torch.from_numpy(tensor_data)
    
    # Fast numpy operations
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    scale = (tensor_max - tensor_min) / 255.0
    zero_point = (-tensor_min / scale).round()
    
    # Vectorized quantization
    quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
    scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
    
    return quantized, scale_zp

def fast_quantize_int4(tensor_data):
    """Fast INT4 quantization using numpy"""
    tensor = torch.from_numpy(tensor_data)
    group_size = 128
    
    # Efficient reshape and pad
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    
    num_groups = (tensor_flat.numel() + group_size - 1) // group_size
    padded_size = num_groups * group_size
    
    if tensor_flat.numel() < padded_size:
        tensor_flat = torch.nn.functional.pad(tensor_flat, (0, padded_size - tensor_flat.numel()))
    
    # Vectorized group processing
    tensor_grouped = tensor_flat.reshape(num_groups, group_size)
    
    # Fast min/max per group
    group_mins = torch.min(tensor_grouped, dim=1)[0]
    group_maxs = torch.max(tensor_grouped, dim=1)[0]
    
    scales = (group_maxs - group_mins) / 15.0
    zero_points = (-group_mins / scales).round()
    
    # Vectorized quantization
    scales_expanded = scales.unsqueeze(1)
    zero_points_expanded = zero_points.unsqueeze(1)
    
    quantized = ((tensor_grouped / scales_expanded) + zero_points_expanded).round().clamp(0, 15).to(torch.uint8)
    
    # Fast packing
    quantized_flat = quantized.flatten()
    packed = torch.zeros((quantized_flat.numel() + 1) // 2, dtype=torch.uint8)
    
    # Vectorized packing
    for i in range(0, quantized_flat.numel(), 2):
        low = quantized_flat[i]
        high = quantized_flat[i + 1] if i + 1 < quantized_flat.numel() else 0
        packed[i // 2] = (high << 4) | low
    
    return packed, scales, zero_points

def process_tensor_parallel(args):
    """Process single tensor - designed for multiprocessing"""
    tensor_name, tensor_data, tensor_shape = args
    
    # Convert to tensor
    tensor = torch.from_numpy(tensor_data).reshape(tensor_shape)
    original_size = tensor.element_size() * tensor.nelement()
    size_mb = original_size / (1024 * 1024)
    
    result = {
        'tensor_name': tensor_name,
        'original_size': original_size,
        'tensors_to_save': {}
    }
    
    if should_quantize_int4(tensor_name, size_mb):
        # Fast INT4
        packed, scales, zero_points = fast_quantize_int4(tensor_data)
        result['tensors_to_save'][tensor_name] = packed
        result['tensors_to_save'][f"{tensor_name}_scales"] = scales
        result['tensors_to_save'][f"{tensor_name}_zero_points"] = zero_points
        result['tensors_to_save'][f"{tensor_name}_original_shape"] = torch.tensor(tensor.shape)
        result['quantized_size'] = packed.element_size() * packed.nelement()
        result['type'] = 'int4'
        
    elif should_quantize_int8(tensor_name, size_mb):
        # Fast INT8
        quantized, scale_zp = fast_quantize_int8(tensor_data)
        result['tensors_to_save'][tensor_name] = quantized
        result['tensors_to_save'][f"{tensor_name}_scale"] = scale_zp
        result['quantized_size'] = quantized.element_size() * quantized.nelement()
        result['type'] = 'int8'
        
    else:
        # FP16 conversion
        fp16_tensor = tensor.to(torch.float16)
        result['tensors_to_save'][tensor_name] = fp16_tensor
        result['quantized_size'] = fp16_tensor.element_size() * fp16_tensor.nelement()
        result['type'] = 'fp16'
    
    return result

class ParallelQuantizer:
    def __init__(self, model_variant: str = '4b'):
        self.variant = model_variant
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💻 Using {cpu_count} CPU cores for parallel processing")
        logger.info(f"📂 Input: {self.model_path}")
        logger.info(f"📂 Output: {self.output_dir}")
        
        self.stats = {
            'total_original_size': 0,
            'total_quantized_size': 0,
            'tensors_processed': 0,
            'int4_count': 0,
            'int8_count': 0,
            'fp16_count': 0
        }
    
    def process_file_parallel(self, file_path: Path) -> dict:
        """Process file with parallel tensor processing"""
        logger.info(f"🔥 Parallel processing {file_path.name}...")
        
        # Load all tensors first
        tensor_jobs = []
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            logger.info(f"  Loading {len(tensor_names)} tensors...")
            
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                tensor_data = tensor.numpy()
                tensor_shape = tensor.shape
                tensor_jobs.append((tensor_name, tensor_data, tensor_shape))
        
        # Process in parallel
        logger.info(f"  Processing {len(tensor_jobs)} tensors in parallel...")
        all_tensors = {}
        
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            # Submit all jobs
            futures = {executor.submit(process_tensor_parallel, job): job[0] for job in tensor_jobs}
            
            # Collect results
            for future in as_completed(futures):
                tensor_name = futures[future]
                try:
                    result = future.result()
                    
                    # Merge results
                    all_tensors.update(result['tensors_to_save'])
                    
                    # Update stats
                    self.stats['total_original_size'] += result['original_size']
                    self.stats['total_quantized_size'] += result['quantized_size']
                    self.stats['tensors_processed'] += 1
                    
                    if result['type'] == 'int4':
                        self.stats['int4_count'] += 1
                    elif result['type'] == 'int8':
                        self.stats['int8_count'] += 1
                    else:
                        self.stats['fp16_count'] += 1
                        
                    if self.stats['tensors_processed'] % 50 == 0:
                        logger.info(f"  Processed {self.stats['tensors_processed']}/{len(tensor_jobs)} tensors")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {tensor_name}: {e}")
        
        return all_tensors
    
    def quantize_model(self):
        """Parallel quantization"""
        start_time = time.time()
        
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 Parallel quantization of {len(model_files)} files")
        
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"📦 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # Parallel processing
            tensors = self.process_file_parallel(file_path)
            
            # Save results
            output_file = self.output_dir / file_path.name
            metadata = {
                'quantization': 'parallel_cpu_mixed_precision',
                'model_variant': self.variant,
                'cpu_cores_used': cpu_count,
                'unicorn_optimized': 'true',
                'npu_igpu_only': 'true'
            }
            
            save_file(tensors, output_file, metadata=metadata)
            logger.info(f"  💾 Saved {output_file}")
            
            del tensors
            gc.collect()
        
        elapsed = time.time() - start_time
        compression_ratio = self.stats['total_original_size'] / self.stats['total_quantized_size']
        
        logger.info("=" * 60)
        logger.info("🎉 PARALLEL QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Time: {elapsed:.1f} seconds")
        logger.info(f"📊 Tensors: {self.stats['tensors_processed']}")
        logger.info(f"💾 Size: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    quantizer = ParallelQuantizer('4b')
    quantizer.quantize_model()

if __name__ == "__main__":
    main()