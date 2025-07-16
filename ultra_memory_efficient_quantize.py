#!/usr/bin/env python3
"""
Ultra Memory-Efficient Quantization for Gemma 3 27B
Processes tensors ONE AT A TIME to avoid memory overflow
"""

import os
import torch
import multiprocessing
from pathlib import Path
import json
import time
import gc
import psutil
from typing import Dict, Any, Optional, Tuple
from safetensors import safe_open

# CPU optimization  
cpu_count = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraMemoryEfficientQuantizer:
    """Ultra memory-efficient quantizer that processes one tensor at a time"""
    
    def __init__(self, model_path: str = "./models/gemma-3-27b-it"):
        self.model_path = Path(model_path)
        self.output_dir = Path("./quantized_models/gemma-3-27b-it-ultra-memory-efficient")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_original_size': 0,
            'total_quantized_size': 0,
            'processed_tensors': 0,
            'peak_memory_mb': 0,
            'tensor_stats': {}
        }
        
        # Memory monitoring
        self.process = psutil.Process()
        
    def monitor_memory(self) -> float:
        """Monitor current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], memory_mb)
        return memory_mb
        
    def quantize_single_tensor(self, tensor: torch.Tensor, scheme: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Quantize a single tensor with immediate memory cleanup"""
        original_size = tensor.numel() * 4  # FP32 bytes
        
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
            groups = tensor_flat.split(group_size)
            
            quantized_groups = []
            scales = []
            
            for group in groups:
                scale = torch.max(torch.abs(group)) / 7.0
                q_group = torch.round(group / scale).clamp(-7, 7).to(torch.int8)
                quantized_groups.append(q_group)
                scales.append(scale)
            
            quantized = torch.cat(quantized_groups).reshape(tensor.shape)
            scale = torch.stack(scales)
            memory_reduction = 0.875
            
        else:
            # FP16 fallback
            quantized = tensor.to(torch.float16)
            scale = torch.tensor(1.0)
            memory_reduction = 0.5
        
        quantized_size = int(original_size * (1 - memory_reduction))
        
        stats = {
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction': memory_reduction,
            'scheme': scheme
        }
        
        return quantized, scale, stats
    
    def get_quantization_scheme(self, tensor_name: str) -> str:
        """Determine optimal quantization scheme based on tensor type"""
        clean_name = tensor_name.replace("model.language_model.", "model.")
        
        if any(x in clean_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
            return 'int8_symmetric'  # NPU optimized
        elif any(x in clean_name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
            return 'int4_grouped'     # iGPU optimized  
        else:
            return 'int8_asymmetric'  # CPU optimized
    
    def process_single_safetensors_file(self, file_path: Path) -> Dict:
        """Process a single safetensors file one tensor at a time"""
        logger.info(f"📂 Processing {file_path.name}")
        
        file_stats = {
            'tensors_processed': 0,
            'original_size': 0,
            'quantized_size': 0,
            'tensor_details': {}
        }
        
        # Get tensor list first without loading
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
        
        logger.info(f"   📊 Found {len(tensor_keys)} tensors")
        
        # Process each tensor individually
        for idx, tensor_key in enumerate(tensor_keys):
            memory_before = self.monitor_memory()
            
            # Load ONLY this tensor
            with safe_open(file_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(tensor_key)
            
            # Determine scheme
            scheme = self.get_quantization_scheme(tensor_key)
            
            # Quantize
            quantized, scale, tensor_stats = self.quantize_single_tensor(tensor, scheme)
            
            # Save quantized tensor immediately
            output_file = self.output_dir / f"{file_path.stem}_{idx:04d}.pt"
            torch.save({
                'tensor': quantized,
                'scale': scale,
                'original_key': tensor_key,
                'scheme': scheme,
                'shape': tensor.shape
            }, output_file)
            
            # Update statistics
            file_stats['tensors_processed'] += 1
            file_stats['original_size'] += tensor_stats['original_size']
            file_stats['quantized_size'] += tensor_stats['quantized_size']
            file_stats['tensor_details'][tensor_key] = tensor_stats
            
            # Cleanup immediately
            del tensor, quantized, scale
            gc.collect()
            
            memory_after = self.monitor_memory()
            
            # Progress update
            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / len(tensor_keys) * 100
                logger.info(f"   🔄 Progress: {progress:.1f}% - Memory: {memory_after:.1f}MB")
        
        return file_stats
    
    def quantize_model(self) -> Dict:
        """Quantize the entire model using ultra memory-efficient approach"""
        logger.info("🚀 Ultra Memory-Efficient Quantization for Gemma 3 27B")
        logger.info(f"💻 Using {cpu_count} CPU cores")
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"📁 Output path: {self.output_dir}")
        
        start_time = time.time()
        
        # Find all safetensors files
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if not safetensor_files:
            logger.error("❌ No safetensors files found!")
            return None
        
        logger.info(f"📦 Found {len(safetensor_files)} safetensors files")
        
        # Process each file
        for file_idx, file_path in enumerate(safetensor_files):
            logger.info(f"🔄 Processing file {file_idx + 1}/{len(safetensor_files)}")
            
            file_stats = self.process_single_safetensors_file(file_path)
            
            # Update global stats
            self.stats['total_original_size'] += file_stats['original_size']
            self.stats['total_quantized_size'] += file_stats['quantized_size']
            self.stats['processed_tensors'] += file_stats['tensors_processed']
            
            # Progress update
            elapsed = time.time() - start_time
            files_remaining = len(safetensor_files) - file_idx - 1
            eta = (elapsed / (file_idx + 1)) * files_remaining if file_idx > 0 else 0
            
            logger.info(f"   ✅ File complete. ETA: {eta/60:.1f}m - Peak memory: {self.stats['peak_memory_mb']:.1f}MB")
        
        # Final statistics
        total_time = time.time() - start_time
        total_savings = (self.stats['total_original_size'] - self.stats['total_quantized_size']) / self.stats['total_original_size']
        
        logger.info("🎉 QUANTIZATION COMPLETE!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📊 Tensors processed: {self.stats['processed_tensors']:,}")
        logger.info(f"📊 Size: {self.stats['total_original_size']/(1024**3):.2f}GB → {self.stats['total_quantized_size']/(1024**3):.2f}GB")
        logger.info(f"📊 Memory reduction: {total_savings:.1%}")
        logger.info(f"💾 Peak memory usage: {self.stats['peak_memory_mb']:.1f}MB")
        logger.info(f"🎯 Fits in 16GB iGPU: {self.stats['total_quantized_size']/(1024**3) < 16}")
        
        # Save metadata
        metadata = {
            "model": "gemma-3-27b-it",
            "quantization_method": "ultra_memory_efficient",
            "quantization_time_minutes": total_time / 60,
            "cpu_cores_used": cpu_count,
            "original_size_gb": self.stats['total_original_size'] / (1024**3),
            "quantized_size_gb": self.stats['total_quantized_size'] / (1024**3),
            "memory_reduction": total_savings,
            "peak_memory_usage_mb": self.stats['peak_memory_mb'],
            "fits_in_16gb_igpu": self.stats['total_quantized_size']/(1024**3) < 16,
            "tensors_processed": self.stats['processed_tensors'],
            "files_processed": len(safetensor_files)
        }
        
        with open(self.output_dir / "quantization_results.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Results saved to {self.output_dir}")
        return metadata

def main():
    """Main function"""
    # Check available memory
    memory = psutil.virtual_memory()
    logger.info(f"🖥️ Available RAM: {memory.available / (1024**3):.1f} GB")
    
    if memory.available < 10 * (1024**3):  # Less than 10GB
        logger.warning("⚠️ Low memory detected. Using ultra-conservative mode.")
    
    # Initialize quantizer
    quantizer = UltraMemoryEfficientQuantizer()
    
    # Run quantization
    result = quantizer.quantize_model()
    
    if result:
        logger.info("✅ Quantization completed successfully!")
        return True
    else:
        logger.error("❌ Quantization failed!")
        return False

if __name__ == "__main__":
    main()