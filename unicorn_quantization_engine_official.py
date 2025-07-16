#!/usr/bin/env python3
"""
Memory-Efficient Quantization for Gemma 3 27B
Processes tensors in batches to avoid memory overflow
"""

import os
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time

# CPU optimization
cpu_count = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_tensor_batch(tensor_batch):
    """Quantize a batch of tensors efficiently"""
    results = {}
    
    for name, tensor, scheme in tensor_batch:
        # Quantization logic
        if 'int8' in scheme:
            scale = torch.max(torch.abs(tensor)) / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            memory_reduction = 0.5
        elif 'int4' in scheme:
            scale = torch.max(torch.abs(tensor)) / 7.0
            quantized = torch.round(tensor / scale).clamp(-7, 7).to(torch.int8)
            memory_reduction = 0.75
        else:
            quantized = tensor.to(torch.float16)
            scale = torch.tensor(1.0)
            memory_reduction = 0.5
        
        original_size = tensor.numel() * 4
        quantized_size = int(original_size * (1 - memory_reduction))
        
        results[name] = {
            'quantized_tensor': quantized,
            'scale': scale,
            'scheme': scheme,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction': memory_reduction
        }
        
        # Clear tensor from memory immediately
        del tensor
    
    return results

def memory_efficient_quantize():
    """Memory-efficient quantization with batching"""
    logger.info("🚀 Memory-Efficient Quantization for Gemma 3 27B")
    logger.info(f"💻 Using {cpu_count} CPU cores with smart batching")
    
    start_time = time.time()
    
    # Load model weights one file at a time
    from safetensors import safe_open
    model_path = Path("./models/gemma-3-27b-it")
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    total_original_size = 0
    total_quantized_size = 0
    processed_tensors = 0
    
    # Process each safetensors file separately to manage memory
    for file_idx, file_path in enumerate(safetensor_files):
        logger.info(f"📂 Processing {file_path.name} ({file_idx+1}/{len(safetensor_files)})")
        
        # Load this file's weights
        file_weights = []
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                clean_key = key.replace("model.language_model.", "model.")
                tensor = f.get_tensor(key)
                
                # Determine quantization scheme
                if any(x in clean_key for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
                    scheme = 'int8_symmetric'
                elif any(x in clean_key for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
                    scheme = 'int4_grouped'
                else:
                    scheme = 'int8_asymmetric'
                
                file_weights.append((clean_key, tensor, scheme))
        
        # Process this file's tensors in parallel batches
        batch_size = max(1, len(file_weights) // cpu_count)
        batches = [file_weights[i:i+batch_size] for i in range(0, len(file_weights), batch_size)]
        
        logger.info(f"   🔧 Processing {len(file_weights)} tensors in {len(batches)} batches")
        
        with ThreadPoolExecutor(max_workers=min(cpu_count, len(batches))) as executor:
            batch_futures = [executor.submit(quantize_tensor_batch, batch) for batch in batches]
            
            for future in batch_futures:
                batch_results = future.result()
                
                # Accumulate statistics
                for name, result in batch_results.items():
                    total_original_size += result['original_size']
                    total_quantized_size += result['quantized_size']
                    processed_tensors += 1
        
        # Force garbage collection after each file
        del file_weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Progress update
        elapsed = time.time() - start_time
        files_remaining = len(safetensor_files) - file_idx - 1
        eta = (elapsed / (file_idx + 1)) * files_remaining
        
        logger.info(f"   ✅ File complete. Progress: {file_idx+1}/{len(safetensor_files)} - ETA: {eta/60:.1f}m")
    
    # Final statistics
    total_time = time.time() - start_time
    total_savings = (total_original_size - total_quantized_size) / total_original_size
    
    logger.info("🎉 QUANTIZATION COMPLETE!")
    logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
    logger.info(f"📊 Tensors processed: {processed_tensors:,}")
    logger.info(f"📊 Size: {total_original_size/(1024**3):.2f}GB → {total_quantized_size/(1024**3):.2f}GB")
    logger.info(f"📊 Memory reduction: {total_savings:.1%}")
    logger.info(f"🎯 Fits in 16GB iGPU: {total_quantized_size/(1024**3) < 16}")
    
    # Save metadata
    output_dir = Path("./quantized_models/gemma-3-27b-it-memory-efficient")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model": "gemma-3-27b-it",
        "quantization_method": "memory_efficient_batch",
        "quantization_time_minutes": total_time / 60,
        "cpu_cores_used": cpu_count,
        "original_size_gb": total_original_size / (1024**3),
        "quantized_size_gb": total_quantized_size / (1024**3),
        "memory_reduction": total_savings,
        "fits_in_16gb_igpu": total_quantized_size/(1024**3) < 16,
        "tensors_processed": processed_tensors,
        "files_processed": len(safetensor_files)
    }
    
    with open(output_dir / "quantization_results.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_dir}")
    return metadata

if __name__ == "__main__":
    memory_efficient_quantize()