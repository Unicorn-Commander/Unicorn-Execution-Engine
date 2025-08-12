#!/usr/bin/env python3
"""
Universal Gemma-3 Quantizer - Handles 4B, 9B, 12B, and 27B variants
Optimized for NPU+iGPU execution with custom quantization per model size
"""

import os
import torch
import multiprocessing
from pathlib import Path
import logging
import time
from typing import Dict, Any, Tuple
from safetensors import safe_open
from safetensors.torch import save_file
import numpy as np
import gc

# Use ALL CPU cores
cpu_count = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalGemma3Quantizer:
    """Universal quantizer for all Gemma-3 model sizes"""
    
    # Model configurations
    MODEL_CONFIGS = {
        '4b': {
            'layers': 32,
            'hidden_size': 3072,
            'attention_heads': 16,
            'kv_heads': 16,  # No GQA in 4B
            'intermediate_size': 16384,
            'vocab_size': 256128,
            'expected_size_gb': 8,
            'target_size_gb': 2.5
        },
        '9b': {
            'layers': 42,
            'hidden_size': 3584,
            'attention_heads': 16,
            'kv_heads': 8,   # GQA enabled
            'intermediate_size': 14336,
            'vocab_size': 256128,
            'expected_size_gb': 18,
            'target_size_gb': 5
        },
        '12b': {
            'layers': 36,
            'hidden_size': 4096,
            'attention_heads': 32,
            'kv_heads': 16,  # GQA enabled
            'intermediate_size': 22016,
            'vocab_size': 256128,
            'expected_size_gb': 24,
            'target_size_gb': 7
        },
        '27b': {
            'layers': 62,
            'hidden_size': 5376,
            'attention_heads': 32,
            'kv_heads': 16,  # GQA enabled
            'intermediate_size': 21504,
            'vocab_size': 262208,
            'expected_size_gb': 54,
            'target_size_gb': 15
        }
    }
    
    def __init__(self, model_variant: str = '4b'):
        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model variant: {model_variant}. Choose from: {list(self.MODEL_CONFIGS.keys())}")
        
        self.variant = model_variant
        self.config = self.MODEL_CONFIGS[model_variant]
        
        # Paths
        self.model_path = Path(f"/home/ucadmin/Development/AI-Models/gemma-3-{model_variant}-it")
        if not self.model_path.exists():
            # Try alternative path
            self.model_path = Path(f"./models/gemma-3-{model_variant}-it")
        
        self.output_dir = Path(f"./quantized_models/gemma-3-{model_variant}-it-quantized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🦄 Universal Gemma-3 Quantizer - {model_variant.upper()} variant")
        logger.info(f"📂 Input: {self.model_path}")
        logger.info(f"📂 Output: {self.output_dir}")
        logger.info(f"🎯 Target size: {self.config['target_size_gb']}GB (from ~{self.config['expected_size_gb']}GB)")
        
        self.stats = {
            'total_original_size': 0,
            'total_quantized_size': 0,
            'tensors_processed': 0,
            'int4_count': 0,
            'int8_count': 0,
            'fp16_count': 0
        }
    
    def should_quantize_int4(self, tensor_name: str, tensor_size_mb: float) -> bool:
        """Determine if tensor should be INT4 quantized"""
        # FFN weights get INT4 for iGPU
        if any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj']):
            return tensor_size_mb > 1.0  # INT4 for FFN >1MB
        return False
    
    def should_quantize_int8(self, tensor_name: str, tensor_size_mb: float) -> bool:
        """Determine if tensor should be INT8 quantized"""
        # Attention weights get INT8 for NPU
        if any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return True
        # Large embeddings also get INT8
        if 'embed_tokens' in tensor_name and tensor_size_mb > 10:
            return True
        return False
    
    def quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """INT8 asymmetric quantization for NPU"""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = (-tensor_min / scale).round()
        
        quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).to(torch.uint8)
        
        # Store scale and zero_point together
        scale_zp = torch.tensor([scale.item(), zero_point.item()], dtype=torch.float32)
        
        return quantized, scale_zp
    
    def quantize_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """INT4 grouped quantization for iGPU"""
        # Group size for INT4 (e.g., 128 elements per group)
        group_size = 128
        
        # Reshape for grouping
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Pad if necessary
        num_groups = (tensor_flat.numel() + group_size - 1) // group_size
        padded_size = num_groups * group_size
        if tensor_flat.numel() < padded_size:
            tensor_flat = torch.nn.functional.pad(tensor_flat, (0, padded_size - tensor_flat.numel()))
        
        # Reshape into groups
        tensor_grouped = tensor_flat.reshape(num_groups, group_size)
        
        # Quantize each group
        scales = torch.zeros(num_groups, dtype=torch.float32)
        zero_points = torch.zeros(num_groups, dtype=torch.float32)
        
        for i in range(num_groups):
            group = tensor_grouped[i]
            group_min = group.min()
            group_max = group.max()
            
            scale = (group_max - group_min) / 15.0  # 4-bit = 0-15
            zero_point = (-group_min / scale).round()
            
            scales[i] = scale
            zero_points[i] = zero_point
        
        # Quantize
        quantized = torch.zeros_like(tensor_grouped, dtype=torch.uint8)
        for i in range(num_groups):
            group = tensor_grouped[i]
            scale = scales[i]
            zero_point = zero_points[i]
            
            q_group = ((group / scale) + zero_point).round().clamp(0, 15).to(torch.uint8)
            quantized[i] = q_group
        
        # Pack INT4 values (2 per byte)
        quantized_flat = quantized.flatten()
        packed = torch.zeros((quantized_flat.numel() + 1) // 2, dtype=torch.uint8)
        
        for i in range(0, quantized_flat.numel(), 2):
            low = quantized_flat[i]
            high = quantized_flat[i + 1] if i + 1 < quantized_flat.numel() else 0
            packed[i // 2] = (high << 4) | low
        
        return packed, scales, zero_points
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single safetensor file"""
        logger.info(f"📄 Processing {file_path.name}...")
        
        tensors_to_save = {}
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                original_size = tensor.element_size() * tensor.nelement()
                size_mb = original_size / (1024 * 1024)
                
                self.stats['total_original_size'] += original_size
                
                # Skip if it's a scale tensor
                if tensor_name.endswith('_scale') or tensor_name.endswith('_zero_point'):
                    continue
                
                # Decide quantization strategy
                if self.should_quantize_int4(tensor_name, size_mb):
                    # INT4 for FFN (iGPU)
                    logger.info(f"  🔥 INT4: {tensor_name} ({size_mb:.1f}MB)")
                    packed, scales, zero_points = self.quantize_int4(tensor)
                    
                    tensors_to_save[tensor_name] = packed
                    tensors_to_save[f"{tensor_name}_scales"] = scales
                    tensors_to_save[f"{tensor_name}_zero_points"] = zero_points
                    tensors_to_save[f"{tensor_name}_original_shape"] = torch.tensor(tensor.shape)
                    
                    quantized_size = packed.element_size() * packed.nelement()
                    self.stats['int4_count'] += 1
                    
                elif self.should_quantize_int8(tensor_name, size_mb):
                    # INT8 for Attention (NPU)
                    logger.info(f"  ⚡ INT8: {tensor_name} ({size_mb:.1f}MB)")
                    quantized, scale_zp = self.quantize_int8(tensor)
                    
                    tensors_to_save[tensor_name] = quantized
                    tensors_to_save[f"{tensor_name}_scale"] = scale_zp
                    
                    quantized_size = quantized.element_size() * quantized.nelement()
                    self.stats['int8_count'] += 1
                    
                else:
                    # Keep as FP16 for small weights
                    logger.info(f"  ✅ FP16: {tensor_name} ({size_mb:.1f}MB)")
                    fp16_tensor = tensor.to(torch.float16)
                    tensors_to_save[tensor_name] = fp16_tensor
                    
                    quantized_size = fp16_tensor.element_size() * fp16_tensor.nelement()
                    self.stats['fp16_count'] += 1
                
                self.stats['total_quantized_size'] += quantized_size
                self.stats['tensors_processed'] += 1
                
                # Clear memory
                del tensor
                gc.collect()
        
        return tensors_to_save
    
    def quantize_model(self):
        """Main quantization process"""
        start_time = time.time()
        
        # Find all safetensor files
        model_files = list(self.model_path.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No safetensor files found in {self.model_path}")
        
        logger.info(f"🚀 Found {len(model_files)} files to process")
        logger.info(f"💾 Using {cpu_count} CPU cores")
        
        # Process each file
        for idx, file_path in enumerate(sorted(model_files)):
            logger.info(f"\n📦 File {idx+1}/{len(model_files)}: {file_path.name}")
            
            # Process tensors
            tensors = self.process_file(file_path)
            
            # Save quantized tensors
            output_file = self.output_dir / file_path.name
            
            # Add metadata
            metadata = {
                'quantization': 'mixed_int4_int8_fp16',
                'model_variant': self.variant,
                'unicorn_optimized': 'true',
                'npu_igpu_only': 'true'
            }
            
            # Add per-tensor metadata
            for name, tensor in tensors.items():
                if name.endswith('_scale') or name.endswith('_scales') or name.endswith('_zero_points'):
                    continue
                    
                # Determine quantization type
                if f"{name}_scales" in tensors:
                    metadata[name] = 'int4_grouped'
                elif f"{name}_scale" in tensors:
                    metadata[name] = 'int8_asymmetric'
                else:
                    metadata[name] = 'fp16'
            
            save_file(tensors, output_file, metadata=metadata)
            logger.info(f"  💾 Saved to {output_file}")
            
            # Clear memory
            del tensors
            gc.collect()
        
        # Report results
        elapsed = time.time() - start_time
        compression_ratio = self.stats['total_original_size'] / self.stats['total_quantized_size']
        
        logger.info("\n" + "="*60)
        logger.info("🎉 QUANTIZATION COMPLETE!")
        logger.info(f"⏱️  Time: {elapsed:.1f} seconds")
        logger.info(f"📊 Tensors processed: {self.stats['tensors_processed']}")
        logger.info(f"   - INT4 (iGPU): {self.stats['int4_count']}")
        logger.info(f"   - INT8 (NPU): {self.stats['int8_count']}")
        logger.info(f"   - FP16: {self.stats['fp16_count']}")
        logger.info(f"💾 Size reduction: {self.stats['total_original_size']/1e9:.1f}GB → {self.stats['total_quantized_size']/1e9:.1f}GB")
        logger.info(f"🔥 Compression ratio: {compression_ratio:.1f}x")
        logger.info(f"✅ Output: {self.output_dir}")

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Universal Gemma-3 Quantizer')
    parser.add_argument('--variant', choices=['4b', '9b', '12b', '27b'], 
                       default='4b', help='Model variant to quantize')
    args = parser.parse_args()
    
    quantizer = UniversalGemma3Quantizer(args.variant)
    quantizer.quantize_model()

if __name__ == "__main__":
    main()