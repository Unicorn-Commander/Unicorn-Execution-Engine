#!/usr/bin/env python3
"""
Fast Quantizer for Gemma-3-4B - Much faster than original
Uses vectorized operations and efficient memory handling
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

# Maximize CPU usage
cpu_count = mp.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastQuantizer:
    def __init__(self, model_variant: str = '4b'):
        self.variant = model_variant
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"⚡ Fast quantizer using {cpu_count} CPU cores")
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
    
    def should_quantize_int4(self, tensor_name: str, tensor_size_mb: float) -> bool:
        """FFN weights get INT4 for iGPU"""
        return any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj']) and tensor_size_mb > 1.0
    
    def should_quantize_int8(self, tensor_name: str, tensor_size_mb: float) -> bool:
        """Attention weights get INT8 for NPU"""
        if any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return True
        if 'embed_tokens' in tensor_name and tensor_size_mb > 10:
            return True
        return False
    
    def fast_quantize_int8(self, tensor: torch.Tensor) -> tuple:
        """Ultra-fast INT8 quantization"""
        # Convert to float32 for processing
        tensor = tensor.float()
        
        # Vectorized operations
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = (-tensor_min / scale).round()
        
        # Fast vectorized quantization
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
        scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
        
        return quantized, scale_zp
    
    def fast_quantize_int4(self, tensor: torch.Tensor) -> tuple:
        """Ultra-fast INT4 quantization"""
        # Convert to float32 for processing
        tensor = tensor.float()
        
        # Simplified INT4 - no grouping for speed
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 15.0
        zero_point = (-tensor_min / scale).round()
        
        # Fast quantization
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 15).to(torch.uint8)
        
        # Simple packing (2 values per byte)
        flat = quantized.flatten()
        packed = torch.zeros((flat.numel() + 1) // 2, dtype=torch.uint8)
        
        # Efficient packing
        for i in range(0, flat.numel(), 2):
            low = flat[i]
            high = flat[i + 1] if i + 1 < flat.numel() else 0
            packed[i // 2] = (high << 4) | low
        
        return packed, scale, zero_point
    
    def process_file_fast(self, file_path: Path) -> dict:
        """Ultra-fast file processing"""
        logger.info(f"⚡ Fast processing {file_path.name}...")
        
        tensors_to_save = {}
        processed_count = 0
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            total_tensors = len(tensor_names)
            
            for tensor_name in tensor_names:
                # Handle BFloat16 conversion
                tensor = f.get_tensor(tensor_name)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                
                original_size = tensor.element_size() * tensor.nelement()
                size_mb = original_size / (1024 * 1024)
                
                self.stats['total_original_size'] += original_size
                
                if self.should_quantize_int4(tensor_name, size_mb):
                    # Fast INT4
                    packed, scale, zero_point = self.fast_quantize_int4(tensor)
                    tensors_to_save[tensor_name] = packed
                    tensors_to_save[f"{tensor_name}_scale"] = scale
                    tensors_to_save[f"{tensor_name}_zero_point"] = zero_point
                    tensors_to_save[f"{tensor_name}_original_shape"] = torch.tensor(tensor.shape)
                    
                    quantized_size = packed.element_size() * packed.nelement()
                    self.stats['int4_count'] += 1
                    logger.info(f"  🔥 INT4: {tensor_name} ({size_mb:.1f}MB)")
                    
                elif self.should_quantize_int8(tensor_name, size_mb):
                    # Fast INT8
                    quantized, scale_zp = self.fast_quantize_int8(tensor)
                    tensors_to_save[tensor_name] = quantized
                    tensors_to_save[f"{tensor_name}_scale"] = scale_zp
                    
                    quantized_size = quantized.element_size() * quantized.nelement()
                    self.stats['int8_count'] += 1
                    logger.info(f"  ⚡ INT8: {tensor_name} ({size_mb:.1f}MB)")
                    
                else:
                    # FP16 conversion
                    fp16_tensor = tensor.to(torch.float16)
                    tensors_to_save[tensor_name] = fp16_tensor
                    
                    quantized_size = fp16_tensor.element_size() * fp16_tensor.nelement()
                    self.stats['fp16_count'] += 1
                    logger.info(f"  ✅ FP16: {tensor_name} ({size_mb:.1f}MB)")
                
                self.stats['total_quantized_size'] += quantized_size
                self.stats['tensors_processed'] += 1
                processed_count += 1
                
                # Progress update
                if processed_count % 20 == 0:
                    logger.info(f"  Progress: {processed_count}/{total_tensors} tensors")
                
                # Memory cleanup
                del tensor
                gc.collect()
        
        return tensors_to_save
    
    def quantize_model(self):
        """Ultra-fast quantization"""
        start_time = time.time()
        
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 Fast quantization of {len(model_files)} files")
        
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"📦 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # Fast processing
            tensors = self.process_file_fast(file_path)
            
            # Save results
            output_file = self.output_dir / file_path.name
            metadata = {
                'quantization': 'fast_mixed_precision',
                'model_variant': str(self.variant),
                'cpu_cores_used': str(cpu_count),
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
        logger.info("🎉 FAST QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Time: {elapsed:.1f} seconds")
        logger.info(f"📊 Tensors: {self.stats['tensors_processed']}")
        logger.info(f"   - INT4 (iGPU): {self.stats['int4_count']}")
        logger.info(f"   - INT8 (NPU): {self.stats['int8_count']}")
        logger.info(f"   - FP16: {self.stats['fp16_count']}")
        logger.info(f"💾 Size: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    quantizer = FastQuantizer('4b')
    quantizer.quantize_model()

if __name__ == "__main__":
    main()