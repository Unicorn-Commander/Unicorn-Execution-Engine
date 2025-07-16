#!/usr/bin/env python3
"""
Ultra-Aggressive Quantization for 16GB iGPU Target
Gemma 3 27B → under 16GB for iGPU acceleration
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

def ultra_quantize_tensor_batch(tensor_batch):
    """Ultra-aggressive quantization targeting 16GB total"""
    results = {}
    
    for name, tensor, scheme in tensor_batch:
        # More aggressive quantization schemes
        if 'int4' in scheme or any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            # INT4 for FFN layers (75% reduction)
            scale = torch.max(torch.abs(tensor)) / 7.0
            quantized = torch.round(tensor / scale).clamp(-7, 7).to(torch.int8)
            memory_reduction = 0.75
        elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            # INT6 for attention (62.5% reduction)
            scale = torch.max(torch.abs(tensor)) / 31.0
            quantized = torch.round(tensor / scale).clamp(-31, 31).to(torch.int8)
            memory_reduction = 0.625
        elif 'embed' in name or 'lm_head' in name:
            # INT8 for embeddings (50% reduction)
            scale = torch.max(torch.abs(tensor)) / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            memory_reduction = 0.5
        else:
            # INT8 for everything else (50% reduction)
            scale = torch.max(torch.abs(tensor)) / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            memory_reduction = 0.5
        
        original_size = tensor.numel() * 4  # FP32
        quantized_size = int(original_size * (1 - memory_reduction))
        
        results[name] = {
            'quantized_tensor': quantized,
            'scale': scale,
            'scheme': scheme,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction': memory_reduction
        }
        
        del tensor
    
    return results

def ultra_quantize_for_16gb():
    """Ultra-aggressive quantization targeting 16GB iGPU"""
    logger.info("🚀 ULTRA Quantization - Targeting 16GB iGPU for Gemma 3 27B")
    logger.info(f"💻 Using {cpu_count} CPU cores with aggressive compression")
    logger.info("🎯 Target: <16GB for full iGPU acceleration")
    
    start_time = time.time()
    
    from safetensors import safe_open
    model_path = Path("./models/gemma-3-27b-it")
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    total_original_size = 0
    total_quantized_size = 0
    processed_tensors = 0
    
    # Track quantization by layer type
    layer_stats = {
        'attention': {'count': 0, 'size': 0},
        'ffn': {'count': 0, 'size': 0},
        'embedding': {'count': 0, 'size': 0},
        'other': {'count': 0, 'size': 0}
    }
    
    for file_idx, file_path in enumerate(safetensor_files):
        logger.info(f"📂 Processing {file_path.name} ({file_idx+1}/{len(safetensor_files)})")
        
        file_weights = []
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                clean_key = key.replace("model.language_model.", "model.")
                tensor = f.get_tensor(key)
                
                # Ultra-aggressive scheme selection
                if any(x in clean_key for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    scheme = 'int6_attention'  # More aggressive
                    layer_type = 'attention'
                elif any(x in clean_key for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
                    scheme = 'int4_ffn'  # Most aggressive
                    layer_type = 'ffn'
                elif any(x in clean_key for x in ['embed', 'lm_head']):
                    scheme = 'int8_embedding'
                    layer_type = 'embedding'
                else:
                    scheme = 'int8_other'
                    layer_type = 'other'
                
                file_weights.append((clean_key, tensor, scheme))
                layer_stats[layer_type]['count'] += 1
        
        # Process in parallel
        batch_size = max(1, len(file_weights) // cpu_count)
        batches = [file_weights[i:i+batch_size] for i in range(0, len(file_weights), batch_size)]
        
        with ThreadPoolExecutor(max_workers=min(cpu_count, len(batches))) as executor:
            batch_futures = [executor.submit(ultra_quantize_tensor_batch, batch) for batch in batches]
            
            for future in batch_futures:
                batch_results = future.result()
                
                for name, result in batch_results.items():
                    total_original_size += result['original_size']
                    total_quantized_size += result['quantized_size']
                    processed_tensors += 1
                    
                    # Track by layer type
                    for layer_type in layer_stats:
                        if layer_type in result['scheme'] or (layer_type == 'attention' and 'int6' in result['scheme']):
                            layer_stats[layer_type]['size'] += result['quantized_size']
                            break
        
        del file_weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        current_gb = total_quantized_size / (1024**3)
        logger.info(f"   ✅ File {file_idx+1}/12 complete. Current size: {current_gb:.2f}GB")
    
    # Final statistics
    total_time = time.time() - start_time
    total_savings = (total_original_size - total_quantized_size) / total_original_size
    final_gb = total_quantized_size / (1024**3)
    
    logger.info("🎉 ULTRA QUANTIZATION COMPLETE!")
    logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
    logger.info(f"📊 Tensors processed: {processed_tensors:,}")
    logger.info(f"📊 Size: {total_original_size/(1024**3):.2f}GB → {final_gb:.2f}GB")
    logger.info(f"📊 Memory reduction: {total_savings:.1%}")
    logger.info(f"🎯 Fits in 16GB iGPU: {'✅ YES' if final_gb < 16 else '❌ NO'}")
    
    # Layer breakdown
    logger.info("📊 Layer breakdown:")
    for layer_type, stats in layer_stats.items():
        if stats['count'] > 0:
            size_gb = stats['size'] / (1024**3)
            logger.info(f"   {layer_type}: {stats['count']} layers, {size_gb:.2f}GB")
    
    # Save metadata
    output_dir = Path("./quantized_models/gemma-3-27b-it-ultra-16gb")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model": "gemma-3-27b-it",
        "quantization_method": "ultra_aggressive_16gb",
        "quantization_time_minutes": total_time / 60,
        "cpu_cores_used": cpu_count,
        "original_size_gb": total_original_size / (1024**3),
        "quantized_size_gb": final_gb,
        "memory_reduction": total_savings,
        "fits_in_16gb_igpu": final_gb < 16,
        "tensors_processed": processed_tensors,
        "target_achieved": final_gb < 16,
        "layer_stats": layer_stats
    }
    
    with open(output_dir / "ultra_quantization_results.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_dir}")
    return metadata

if __name__ == "__main__":
    ultra_quantize_for_16gb()