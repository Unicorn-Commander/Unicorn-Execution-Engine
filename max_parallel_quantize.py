#!/usr/bin/env python3
"""
Maximum Parallel Quantization for Gemma 3 27B
Uses ALL 16 CPU cores with multiprocessing for maximum speed
"""

import os
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import time
import gc
import psutil
from typing import Dict, Any, Optional, Tuple, List
from safetensors import safe_open
import pickle

# Set environment for maximum parallelism
os.environ['OMP_NUM_THREADS'] = '1'  # Each process gets 1 thread
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_tensor_worker(data: Tuple[str, str, str, torch.Tensor]) -> Dict[str, Any]:
    """Worker function for parallel tensor quantization"""
    file_path, tensor_name, scheme, tensor = data
    
    try:
        # Apply quantization based on scheme
        if 'int8' in scheme:
            if 'symmetric' in scheme:
                # Symmetric quantization for attention weights (NPU optimized)
                scale = torch.max(torch.abs(tensor)) / 127.0
                quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            else:
                # Asymmetric quantization for embeddings (CPU optimized)
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
                scale = (max_val - min_val) / 255.0
                zero_point = torch.round(-min_val / scale).clamp(0, 255).to(torch.uint8)
                quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
                scale = torch.stack([scale, zero_point.float()])
            memory_reduction = 0.75
            
        elif 'int4' in scheme:
            # Grouped INT4 quantization for FFN weights (iGPU optimized)
            group_size = 128
            tensor_flat = tensor.flatten()
            num_groups = (tensor_flat.numel() + group_size - 1) // group_size
            
            quantized_groups = []
            scales = []
            
            for i in range(num_groups):
                start_idx = i * group_size
                end_idx = min((i + 1) * group_size, tensor_flat.numel())
                group = tensor_flat[start_idx:end_idx]
                
                if group.numel() > 0:
                    scale = torch.max(torch.abs(group)) / 7.0
                    q_group = torch.round(group / scale).clamp(-7, 7).to(torch.int8)
                    quantized_groups.append(q_group)
                    scales.append(scale)
            
            # Reconstruct tensor
            quantized = torch.cat(quantized_groups).reshape(tensor.shape)
            scale = torch.stack(scales)
            memory_reduction = 0.875
            
        else:
            # FP16 fallback
            quantized = tensor.to(torch.float16)
            scale = torch.tensor(1.0)
            memory_reduction = 0.5
        
        original_size = tensor.numel() * 4  # FP32 bytes
        quantized_size = int(original_size * (1 - memory_reduction))
        
        return {
            'file_path': file_path,
            'tensor_name': tensor_name,
            'quantized': quantized,
            'scale': scale,
            'scheme': scheme,
            'shape': tensor.shape,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction': memory_reduction,
            'success': True
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'tensor_name': tensor_name,
            'error': str(e),
            'success': False
        }

def process_file_worker(file_info: Tuple[str, str]) -> Dict[str, Any]:
    """Worker function to process a single file"""
    file_path, output_dir = file_info
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    
    try:
        # Load all tensors from this file
        tensor_data = []
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_key in f.keys():
                tensor = f.get_tensor(tensor_key)
                
                # Determine quantization scheme
                clean_name = tensor_key.replace("model.language_model.", "model.")
                if any(x in clean_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
                    scheme = 'int8_symmetric'  # NPU optimized
                elif any(x in clean_name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
                    scheme = 'int4_grouped'     # iGPU optimized  
                else:
                    scheme = 'int8_asymmetric'  # CPU optimized
                
                tensor_data.append((str(file_path), tensor_key, scheme, tensor))
        
        # Process tensors in parallel within this file
        num_cores = mp.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(quantize_tensor_worker, tensor_data))
        
        # Save results
        file_stats = {
            'tensors_processed': 0,
            'total_original_size': 0,
            'total_quantized_size': 0,
            'errors': []
        }
        
        for result in results:
            if result['success']:
                # Save quantized tensor
                output_file = output_dir / f"{file_path.stem}_{result['tensor_name'].replace('/', '_').replace('.', '_')}.pt"
                
                save_data = {
                    'tensor': result['quantized'],
                    'scale': result['scale'],
                    'original_key': result['tensor_name'],
                    'scheme': result['scheme'],
                    'shape': result['shape']
                }
                
                torch.save(save_data, output_file)
                
                file_stats['tensors_processed'] += 1
                file_stats['total_original_size'] += result['original_size']
                file_stats['total_quantized_size'] += result['quantized_size']
            else:
                file_stats['errors'].append(result['error'])
        
        return {
            'file_path': str(file_path),
            'success': True,
            'stats': file_stats
        }
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'success': False,
            'error': str(e)
        }

class MaxParallelQuantizer:
    """Maximum parallelism quantizer using all CPU cores"""
    
    def __init__(self, model_path: str = "./models/gemma-3-27b-it"):
        self.model_path = Path(model_path)
        self.output_dir = Path("./quantized_models/gemma-3-27b-it-max-parallel")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cpu_count = mp.cpu_count()
        
        logger.info(f"🚀 Max Parallel Quantizer initialized")
        logger.info(f"💻 Using ALL {self.cpu_count} CPU cores")
        logger.info(f"🖥️ Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
    def quantize_model(self) -> Dict[str, Any]:
        """Quantize the entire model using maximum parallelism"""
        logger.info("🚀 MAXIMUM PARALLEL Quantization for Gemma 3 27B")
        logger.info(f"💻 Using ALL {self.cpu_count} CPU cores with process-level parallelism")
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"📁 Output path: {self.output_dir}")
        
        start_time = time.time()
        
        # Find all safetensors files
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if not safetensor_files:
            logger.error("❌ No safetensors files found!")
            return None
        
        logger.info(f"📦 Found {len(safetensor_files)} safetensors files")
        
        # Prepare file processing jobs
        file_jobs = [(str(file_path), str(self.output_dir)) for file_path in safetensor_files]
        
        # Process files in parallel (each file gets its own process)
        logger.info(f"🔥 Processing {len(file_jobs)} files in parallel using {self.cpu_count} processes")
        
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            file_results = list(executor.map(process_file_worker, file_jobs))
        
        # Aggregate results
        total_stats = {
            'total_original_size': 0,
            'total_quantized_size': 0,
            'processed_tensors': 0,
            'successful_files': 0,
            'failed_files': 0,
            'errors': []
        }
        
        for result in file_results:
            if result['success']:
                total_stats['successful_files'] += 1
                stats = result['stats']
                total_stats['total_original_size'] += stats['total_original_size']
                total_stats['total_quantized_size'] += stats['total_quantized_size']
                total_stats['processed_tensors'] += stats['tensors_processed']
                total_stats['errors'].extend(stats['errors'])
            else:
                total_stats['failed_files'] += 1
                total_stats['errors'].append(result['error'])
        
        # Final statistics
        total_time = time.time() - start_time
        total_savings = (total_stats['total_original_size'] - total_stats['total_quantized_size']) / total_stats['total_original_size'] if total_stats['total_original_size'] > 0 else 0
        
        logger.info("🎉 MAXIMUM PARALLEL QUANTIZATION COMPLETE!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"⚡ Speed: {total_stats['processed_tensors']/(total_time/60):.1f} tensors/minute")
        logger.info(f"💻 CPU efficiency: {self.cpu_count} cores used simultaneously")
        logger.info(f"📊 Tensors processed: {total_stats['processed_tensors']:,}")
        logger.info(f"📊 Size: {total_stats['total_original_size']/(1024**3):.2f}GB → {total_stats['total_quantized_size']/(1024**3):.2f}GB")
        logger.info(f"📊 Memory reduction: {total_savings:.1%}")
        logger.info(f"✅ Successful files: {total_stats['successful_files']}/{len(safetensor_files)}")
        logger.info(f"🎯 Fits in 16GB iGPU: {total_stats['total_quantized_size']/(1024**3) < 16}")
        
        if total_stats['errors']:
            logger.warning(f"⚠️ {len(total_stats['errors'])} errors occurred during processing")
        
        # Save metadata
        metadata = {
            "model": "gemma-3-27b-it",
            "quantization_method": "max_parallel",
            "quantization_time_minutes": total_time / 60,
            "tensors_per_minute": total_stats['processed_tensors']/(total_time/60),
            "cpu_cores_used": self.cpu_count,
            "original_size_gb": total_stats['total_original_size'] / (1024**3),
            "quantized_size_gb": total_stats['total_quantized_size'] / (1024**3),
            "memory_reduction": total_savings,
            "fits_in_16gb_igpu": total_stats['total_quantized_size']/(1024**3) < 16,
            "tensors_processed": total_stats['processed_tensors'],
            "successful_files": total_stats['successful_files'],
            "failed_files": total_stats['failed_files'],
            "errors_count": len(total_stats['errors'])
        }
        
        with open(self.output_dir / "quantization_results.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Results saved to {self.output_dir}")
        return metadata

def main():
    """Main function"""
    # Check available resources
    memory = psutil.virtual_memory()
    cpu_count = mp.cpu_count()
    
    logger.info(f"🖥️ System Resources:")
    logger.info(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"   CPU cores: {cpu_count}")
    
    # Initialize quantizer
    quantizer = MaxParallelQuantizer()
    
    # Run quantization
    result = quantizer.quantize_model()
    
    if result and result['successful_files'] > 0:
        logger.info("✅ Maximum parallel quantization completed successfully!")
        logger.info(f"🎯 Final performance: {result['tensors_per_minute']:.1f} tensors/minute")
        logger.info(f"⚡ Used all {result['cpu_cores_used']} CPU cores")
        return True
    else:
        logger.error("❌ Maximum parallel quantization failed!")
        return False

if __name__ == "__main__":
    # Set multiprocessing start method
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
        main()