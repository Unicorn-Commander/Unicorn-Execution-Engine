#!/usr/bin/env python3
"""
GPU-Accelerated Quantizer for Gemma-3-4B
Uses GPU compute for fast quantization with parallel processing
"""

import torch
import time
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUAcceleratedQuantizer:
    def __init__(self, model_variant: str = '4b'):
        self.variant = model_variant
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use GPU if available, otherwise CPU
        if torch.cuda.is_available():
            logger.info(f"🔥 Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 Using CPU acceleration")
            
        # Paths
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def gpu_quantize_int8(self, tensor: torch.Tensor) -> tuple:
        """GPU-accelerated INT8 quantization"""
        tensor = tensor.to(self.device)
        
        # Fast GPU min/max
        tensor_min = torch.min(tensor)
        tensor_max = torch.max(tensor)
        
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = (-tensor_min / scale).round()
        
        # Vectorized quantization on GPU
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
        scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
        
        return quantized.cpu(), scale_zp
    
    def gpu_quantize_int4(self, tensor: torch.Tensor) -> tuple:
        """GPU-accelerated INT4 grouped quantization"""
        tensor = tensor.to(self.device)
        group_size = 128
        
        # Reshape and pad for vectorized processing
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        num_groups = (tensor_flat.numel() + group_size - 1) // group_size
        padded_size = num_groups * group_size
        
        if tensor_flat.numel() < padded_size:
            tensor_flat = torch.nn.functional.pad(tensor_flat, (0, padded_size - tensor_flat.numel()))
        
        # Vectorized group processing on GPU
        tensor_grouped = tensor_flat.reshape(num_groups, group_size)
        
        # GPU-accelerated min/max per group
        group_mins = torch.min(tensor_grouped, dim=1)[0]
        group_maxs = torch.max(tensor_grouped, dim=1)[0]
        
        scales = (group_maxs - group_mins) / 15.0
        zero_points = (-group_mins / scales).round()
        
        # Vectorized quantization
        scales_expanded = scales.unsqueeze(1)
        zero_points_expanded = zero_points.unsqueeze(1)
        
        quantized = ((tensor_grouped / scales_expanded) + zero_points_expanded).round().clamp(0, 15).to(torch.uint8)
        
        # Pack INT4 values
        quantized_flat = quantized.flatten()
        packed = torch.zeros((quantized_flat.numel() + 1) // 2, dtype=torch.uint8, device=self.device)
        
        # Vectorized packing
        for i in range(0, quantized_flat.numel(), 2):
            low = quantized_flat[i]
            high = quantized_flat[i + 1] if i + 1 < quantized_flat.numel() else 0
            packed[i // 2] = (high << 4) | low
        
        return packed.cpu(), scales.cpu(), zero_points.cpu()
    
    def process_file_gpu(self, file_path: Path) -> dict:
        """GPU-accelerated file processing"""
        logger.info(f"🔥 GPU Processing {file_path.name}...")
        
        tensors_to_save = {}
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
            
            for i, tensor_name in enumerate(tensor_names):
                if i % 10 == 0:
                    logger.info(f"  Progress: {i}/{len(tensor_names)} tensors")
                
                tensor = f.get_tensor(tensor_name)
                original_size = tensor.element_size() * tensor.nelement()
                size_mb = original_size / (1024 * 1024)
                
                self.stats['total_original_size'] += original_size
                
                if self.should_quantize_int4(tensor_name, size_mb):
                    # Fast GPU INT4
                    packed, scales, zero_points = self.gpu_quantize_int4(tensor)
                    tensors_to_save[tensor_name] = packed
                    tensors_to_save[f"{tensor_name}_scales"] = scales
                    tensors_to_save[f"{tensor_name}_zero_points"] = zero_points
                    tensors_to_save[f"{tensor_name}_original_shape"] = torch.tensor(tensor.shape)
                    
                    quantized_size = packed.element_size() * packed.nelement()
                    self.stats['int4_count'] += 1
                    
                elif self.should_quantize_int8(tensor_name, size_mb):
                    # Fast GPU INT8
                    quantized, scale_zp = self.gpu_quantize_int8(tensor)
                    tensors_to_save[tensor_name] = quantized
                    tensors_to_save[f"{tensor_name}_scale"] = scale_zp
                    
                    quantized_size = quantized.element_size() * quantized.nelement()
                    self.stats['int8_count'] += 1
                    
                else:
                    # FP16 conversion
                    fp16_tensor = tensor.to(torch.float16)
                    tensors_to_save[tensor_name] = fp16_tensor
                    
                    quantized_size = fp16_tensor.element_size() * fp16_tensor.nelement()
                    self.stats['fp16_count'] += 1
                
                self.stats['total_quantized_size'] += quantized_size
                self.stats['tensors_processed'] += 1
                
                # Memory cleanup
                del tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        return tensors_to_save
    
    def quantize_model(self):
        """GPU-accelerated quantization"""
        start_time = time.time()
        
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 GPU-accelerated quantization of {len(model_files)} files")
        
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"📦 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # GPU processing
            tensors = self.process_file_gpu(file_path)
            
            # Save results
            output_file = self.output_dir / file_path.name
            metadata = {
                'quantization': 'gpu_accelerated_mixed_precision',
                'model_variant': self.variant,
                'device_used': str(self.device),
                'unicorn_optimized': 'true',
                'npu_igpu_only': 'true'
            }
            
            save_file(tensors, output_file, metadata=metadata)
            logger.info(f"  💾 Saved {output_file}")
            
            del tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        elapsed = time.time() - start_time
        compression_ratio = self.stats['total_original_size'] / self.stats['total_quantized_size']
        
        logger.info("=" * 60)
        logger.info("🎉 GPU-ACCELERATED QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Time: {elapsed:.1f} seconds")
        logger.info(f"📊 Tensors: {self.stats['tensors_processed']}")
        logger.info(f"💾 Size: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    quantizer = GPUAcceleratedQuantizer('4b')
    quantizer.quantize_model()

if __name__ == "__main__":
    main()