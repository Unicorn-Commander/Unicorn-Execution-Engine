#!/usr/bin/env python3
"""
Parallel Batch Quantizer - Actually uses multiple CPU cores
Processes multiple tensors simultaneously for real speedup
"""

import torch
import time
import logging
import multiprocessing as mp
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Configure for parallel processing
cpu_count = mp.cpu_count()
WORKERS = 12  # Use 12 cores

# Each worker gets 1 thread to prevent contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_tensor(args):
    """Process one tensor - designed for parallel execution"""
    tensor_name, tensor_data, tensor_shape, tensor_dtype, quant_type = args
    
    # Ensure single thread per worker
    torch.set_num_threads(1)
    
    try:
        # Reconstruct tensor
        if tensor_dtype == 'bfloat16':
            tensor = torch.from_numpy(tensor_data).view(tensor_shape).float()
        else:
            tensor = torch.from_numpy(tensor_data).view(tensor_shape)
        
        original_size = tensor.element_size() * tensor.nelement()
        
        if quant_type == 'int4':
            # INT4 quantization for FFN
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            scale = (tensor_max - tensor_min) / 15.0
            zero_point = (-tensor_min / scale).round()
            
            quantized = ((tensor - tensor_min) / scale).round().clamp(0, 15).to(torch.uint8)
            
            # Pack INT4
            flat = quantized.flatten()
            packed = torch.zeros((flat.numel() + 1) // 2, dtype=torch.uint8)
            
            # Fast packing
            flat_np = flat.numpy()
            packed_np = packed.numpy()
            for i in range(0, len(flat_np), 2):
                low = flat_np[i]
                high = flat_np[i + 1] if i + 1 < len(flat_np) else 0
                packed_np[i // 2] = (high << 4) | low
            
            return {
                'tensor_name': tensor_name,
                'tensors': {
                    tensor_name: torch.from_numpy(packed_np),
                    f"{tensor_name}_scale": scale,
                    f"{tensor_name}_zero_point": zero_point,
                    f"{tensor_name}_original_shape": torch.tensor(tensor.shape)
                },
                'type': 'int4',
                'original_size': original_size,
                'quantized_size': packed.numel()
            }
            
        elif quant_type == 'int8':
            # INT8 quantization for attention
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = (-tensor_min / scale).round()
            
            quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
            scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
            
            return {
                'tensor_name': tensor_name,
                'tensors': {
                    tensor_name: quantized,
                    f"{tensor_name}_scale": scale_zp
                },
                'type': 'int8',
                'original_size': original_size,
                'quantized_size': quantized.numel()
            }
            
        else:  # fp16
            fp16_tensor = tensor.to(torch.float16)
            return {
                'tensor_name': tensor_name,
                'tensors': {tensor_name: fp16_tensor},
                'type': 'fp16',
                'original_size': original_size,
                'quantized_size': fp16_tensor.element_size() * fp16_tensor.nelement()
            }
            
    except Exception as e:
        logger.error(f"Error processing {tensor_name}: {str(e)}")
        raise

class ParallelBatchQuantizer:
    def __init__(self, model_variant: str = '4b'):
        self.variant = model_variant
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"⚡ Parallel batch quantizer using {WORKERS} CPU cores")
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
    
    def get_quantization_type(self, tensor_name: str, tensor_size_mb: float) -> str:
        """Determine quantization type"""
        if any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj']) and tensor_size_mb > 1.0:
            return 'int4'
        elif any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'int8'
        elif 'embed_tokens' in tensor_name and tensor_size_mb > 10:
            return 'int8'
        else:
            return 'fp16'
    
    def process_file_parallel(self, file_path: Path) -> dict:
        """Process file with true parallel execution"""
        logger.info(f"⚡ Parallel processing {file_path.name}...")
        start_time = time.time()
        
        # Prepare all tensor jobs
        tensor_jobs = []
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            logger.info(f"  Found {len(tensor_names)} tensors")
            
            for i, tensor_name in enumerate(tensor_names):
                tensor = f.get_tensor(tensor_name)
                
                # Handle dtype
                dtype_str = str(tensor.dtype).replace('torch.', '')
                if tensor.dtype == torch.bfloat16:
                    tensor_np = tensor.float().numpy()
                else:
                    tensor_np = tensor.numpy()
                
                size_mb = (tensor.element_size() * tensor.nelement()) / (1024 * 1024)
                quant_type = self.get_quantization_type(tensor_name, size_mb)
                
                tensor_jobs.append((
                    tensor_name,
                    tensor_np,
                    tensor.shape,
                    dtype_str,
                    quant_type
                ))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Prepared {i + 1}/{len(tensor_names)} tensors")
        
        # Process all tensors in parallel
        logger.info(f"  Processing {len(tensor_jobs)} tensors with {WORKERS} workers...")
        all_tensors = {}
        processed = 0
        
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            # Submit all jobs at once
            future_to_name = {
                executor.submit(process_single_tensor, job): job[0] 
                for job in tensor_jobs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                tensor_name = future_to_name[future]
                try:
                    result = future.result()
                    
                    # Merge tensors
                    all_tensors.update(result['tensors'])
                    
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
                    
                    processed += 1
                    if processed % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        logger.info(f"  Processed {processed}/{len(tensor_jobs)} tensors "
                                   f"({rate:.1f} tensors/sec)")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process {tensor_name}: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"  ✅ Completed in {elapsed:.1f}s ({len(tensor_jobs)/elapsed:.1f} tensors/sec)")
        
        return all_tensors
    
    def quantize_model(self):
        """Run parallel quantization"""
        total_start = time.time()
        
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 Starting parallel quantization of {len(model_files)} files")
        
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"\n📁 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # Process with parallel execution
            tensors = self.process_file_parallel(file_path)
            
            # Save results
            output_file = self.output_dir / file_path.name
            metadata = {
                'quantization': 'parallel_mixed_precision',
                'model_variant': str(self.variant),
                'workers_used': str(WORKERS),
                'unicorn_optimized': 'true',
                'npu_igpu_only': 'true'
            }
            
            logger.info(f"  💾 Saving to {output_file}...")
            save_file(tensors, output_file, metadata=metadata)
            logger.info(f"  ✅ Saved successfully")
            
            # Cleanup
            del tensors
            gc.collect()
        
        total_elapsed = time.time() - total_start
        compression_ratio = self.stats['total_original_size'] / self.stats['total_quantized_size']
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PARALLEL QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"⚡ Average speed: {self.stats['tensors_processed']/total_elapsed:.1f} tensors/second")
        logger.info(f"📊 Tensors processed: {self.stats['tensors_processed']}")
        logger.info(f"   - INT4 (iGPU): {self.stats['int4_count']}")
        logger.info(f"   - INT8 (NPU): {self.stats['int8_count']}")
        logger.info(f"   - FP16: {self.stats['fp16_count']}")
        logger.info(f"💾 Size: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression ratio: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    quantizer = ParallelBatchQuantizer('4b')
    quantizer.quantize_model()

if __name__ == "__main__":
    main()