#!/usr/bin/env python3
"""
Batch Quantizer for Gemma-3-4B - Process tensors in parallel batches
Uses all CPU cores efficiently without running out of memory
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

# Use all CPU cores
cpu_count = mp.cpu_count()
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent nested parallelism
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.virtual_memory().used / 1024**3

def quantize_tensor_batch(args):
    """Process a single tensor - optimized for parallel execution"""
    tensor_name, tensor_data, tensor_shape, quantization_type = args
    
    # Set thread count for this process
    torch.set_num_threads(1)
    
    # Convert to tensor
    if isinstance(tensor_data, (bytes, memoryview)):
        tensor = torch.frombuffer(tensor_data, dtype=torch.float32).reshape(tensor_shape)
    else:
        tensor = torch.from_numpy(tensor_data).float()
    
    result = {
        'tensor_name': tensor_name,
        'original_size': tensor.element_size() * tensor.nelement()
    }
    
    if quantization_type == 'int4':
        # Fast INT4 quantization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 15.0
        zero_point = (-tensor_min / scale).round()
        
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 15).to(torch.uint8)
        
        # Pack INT4
        flat = quantized.flatten()
        packed = torch.zeros((flat.numel() + 1) // 2, dtype=torch.uint8)
        
        for i in range(0, flat.numel(), 2):
            low = flat[i]
            high = flat[i + 1] if i + 1 < flat.numel() else 0
            packed[i // 2] = (high << 4) | low
        
        result['tensors'] = {
            tensor_name: packed,
            f"{tensor_name}_scale": scale,
            f"{tensor_name}_zero_point": zero_point,
            f"{tensor_name}_original_shape": torch.tensor(tensor.shape)
        }
        result['quantized_size'] = packed.element_size() * packed.nelement()
        
    elif quantization_type == 'int8':
        # Fast INT8 quantization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = (-tensor_min / scale).round()
        
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
        scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
        
        result['tensors'] = {
            tensor_name: quantized,
            f"{tensor_name}_scale": scale_zp
        }
        result['quantized_size'] = quantized.element_size() * quantized.nelement()
        
    else:  # fp16
        # FP16 conversion
        fp16_tensor = tensor.to(torch.float16)
        result['tensors'] = {tensor_name: fp16_tensor}
        result['quantized_size'] = fp16_tensor.element_size() * fp16_tensor.nelement()
    
    return result

class BatchQuantizer:
    def __init__(self, model_variant: str = '4b', batch_size: int = 16):
        self.variant = model_variant
        self.batch_size = batch_size
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate optimal batch size based on available memory
        available_memory = psutil.virtual_memory().available / 1024**3  # GB
        self.batch_size = min(batch_size, int(available_memory / 4))  # Use 1/4 of available memory
        
        logger.info(f"🚀 Batch quantizer using {cpu_count} CPU cores")
        logger.info(f"📦 Batch size: {self.batch_size} tensors")
        logger.info(f"💾 Available memory: {available_memory:.1f}GB")
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
        """Determine quantization type for tensor"""
        if any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj']) and tensor_size_mb > 1.0:
            return 'int4'
        elif any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'int8'
        elif 'embed_tokens' in tensor_name and tensor_size_mb > 10:
            return 'int8'
        else:
            return 'fp16'
    
    def process_file_batch(self, file_path: Path) -> dict:
        """Process file with batch parallel processing"""
        logger.info(f"📦 Batch processing {file_path.name}...")
        
        # First pass - collect tensor info
        tensor_jobs = []
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            logger.info(f"  Found {len(tensor_names)} tensors")
            
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                
                # Handle BFloat16
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                
                size_mb = (tensor.element_size() * tensor.nelement()) / (1024 * 1024)
                quant_type = self.get_quantization_type(tensor_name, size_mb)
                
                # Convert to numpy for pickling
                tensor_data = tensor.numpy()
                tensor_jobs.append((tensor_name, tensor_data, tensor.shape, quant_type))
        
        # Process in batches with all CPU cores
        logger.info(f"  Processing {len(tensor_jobs)} tensors in batches of {self.batch_size}...")
        all_tensors = {}
        
        with ProcessPoolExecutor(max_workers=12) as executor:
            # Process in batches
            for batch_start in range(0, len(tensor_jobs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(tensor_jobs))
                batch_jobs = tensor_jobs[batch_start:batch_end]
                
                logger.info(f"  Processing batch {batch_start//self.batch_size + 1}/{(len(tensor_jobs) + self.batch_size - 1)//self.batch_size} "
                           f"(tensors {batch_start}-{batch_end-1})")
                
                # Submit batch
                futures = {executor.submit(quantize_tensor_batch, job): job[0] for job in batch_jobs}
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        
                        # Merge results
                        all_tensors.update(result['tensors'])
                        
                        # Update stats
                        self.stats['total_original_size'] += result['original_size']
                        self.stats['total_quantized_size'] += result['quantized_size']
                        self.stats['tensors_processed'] += 1
                        
                        # Count types
                        tensor_name = result['tensor_name']
                        if f"{tensor_name}_scales" in result['tensors']:
                            self.stats['int4_count'] += 1
                        elif f"{tensor_name}_scale" in result['tensors']:
                            self.stats['int8_count'] += 1
                        else:
                            self.stats['fp16_count'] += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing tensor: {e}")
                
                # Memory cleanup after each batch
                gc.collect()
                
                # Log memory usage
                mem_usage = get_memory_usage()
                logger.info(f"  Memory usage: {mem_usage:.1f}GB")
        
        return all_tensors
    
    def quantize_model(self):
        """Batch quantization with parallel processing"""
        start_time = time.time()
        
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 Starting batch quantization of {len(model_files)} files")
        
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"\n📁 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # Batch processing
            tensors = self.process_file_batch(file_path)
            
            # Save results
            output_file = self.output_dir / file_path.name
            metadata = {
                'quantization': 'batch_mixed_precision',
                'model_variant': str(self.variant),
                'cpu_cores_used': str(cpu_count),
                'batch_size': str(self.batch_size),
                'unicorn_optimized': 'true',
                'npu_igpu_only': 'true'
            }
            
            save_file(tensors, output_file, metadata=metadata)
            logger.info(f"  💾 Saved {output_file}")
            
            del tensors
            gc.collect()
        
        elapsed = time.time() - start_time
        compression_ratio = self.stats['total_original_size'] / self.stats['total_quantized_size']
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 BATCH QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"⚡ Speed: {self.stats['tensors_processed']/elapsed:.1f} tensors/second")
        logger.info(f"📊 Tensors: {self.stats['tensors_processed']}")
        logger.info(f"   - INT4 (iGPU): {self.stats['int4_count']}")
        logger.info(f"   - INT8 (NPU): {self.stats['int8_count']}")
        logger.info(f"   - FP16: {self.stats['fp16_count']}")
        logger.info(f"💾 Size: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    # Use 12 cores with larger batches for faster processing
    quantizer = BatchQuantizer('4b', batch_size=24)
    quantizer.quantize_model()

if __name__ == "__main__":
    main()