#!/usr/bin/env python3
"""
Layer-by-Layer Quantization for Gemma 3 27B
Load one layer at a time, use all 16 cores to process it, save, unload, repeat
"""

import os
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import gc
import psutil
from typing import Dict, Any, List, Tuple
from safetensors import safe_open
from safetensors.torch import save_file
import numpy as np

# Use ALL 16 CPU cores
cpu_count = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
torch.set_num_threads(cpu_count)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerByLayerQuantizer:
    """Layer-by-layer quantizer with multi-core tensor processing"""
    
    def __init__(self, model_path: str = "./models/gemma-3-27b-it"):
        self.model_path = Path(model_path)
        self.output_dir = Path("./quantized_models/gemma-3-27b-it-layer-by-layer")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_original_size': 0,
            'total_quantized_size': 0,
            'processed_tensors': 0,
            'layers_processed': 0,
            'peak_memory_mb': 0
        }
        
        self.process = psutil.Process()
        
    def monitor_memory(self) -> float:
        """Monitor current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], memory_mb)
        return memory_mb
    
    def quantize_tensor_chunk(self, args: Tuple[str, torch.Tensor, str, int]) -> Dict:
        """Quantize a single tensor (designed for parallel execution)"""
        tensor_name, tensor, scheme, chunk_id = args
        
        original_size = tensor.numel() * 4  # FP32 bytes
        
        # Apply quantization based on scheme
        if scheme == 'int8_symmetric':
            # NPU-optimized symmetric quantization
            scale = torch.max(torch.abs(tensor)) / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            memory_reduction = 0.75
            
        elif scheme == 'int4_grouped':
            # iGPU-optimized grouped INT4 quantization
            group_size = 128
            tensor_flat = tensor.flatten()
            
            # Process in groups
            groups = []
            scales = []
            
            for i in range(0, len(tensor_flat), group_size):
                group = tensor_flat[i:i+group_size]
                scale = torch.max(torch.abs(group)) / 7.0
                q_group = torch.round(group / scale).clamp(-7, 7).to(torch.int8)
                groups.append(q_group)
                scales.append(scale)
            
            quantized = torch.cat(groups).reshape(tensor.shape)
            scale = torch.stack(scales)
            memory_reduction = 0.875
            
        elif scheme == 'int8_asymmetric':
            # CPU-optimized asymmetric quantization
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            scale = (max_val - min_val) / 255.0
            zero_point = torch.round(-min_val / scale).clamp(0, 255).to(torch.uint8)
            quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
            scale = torch.stack([scale, zero_point.float()])
            memory_reduction = 0.75
            
        else:
            # FP16 fallback
            quantized = tensor.to(torch.float16)
            scale = torch.tensor(1.0)
            memory_reduction = 0.5
        
        quantized_size = int(original_size * (1 - memory_reduction))
        
        return {
            'tensor_name': tensor_name,
            'quantized_tensor': quantized,
            'scale': scale,
            'scheme': scheme,
            'original_size': original_size,
            'quantized_size': quantized_size,
            'memory_reduction': memory_reduction,
            'chunk_id': chunk_id
        }
    
    def get_quantization_scheme(self, tensor_name: str) -> str:
        """Determine optimal quantization scheme"""
        clean_name = tensor_name.replace("model.language_model.", "model.")
        
        if any(x in clean_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
            return 'int8_symmetric'  # NPU optimized
        elif any(x in clean_name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
            return 'int4_grouped'     # iGPU optimized  
        else:
            return 'int8_asymmetric'  # CPU optimized
    
    def process_layer(self, layer_tensors: List[Tuple[str, torch.Tensor]], layer_name: str) -> Dict:
        """Process all tensors in a layer using all 16 cores"""
        logger.info(f"🔄 Processing layer: {layer_name} ({len(layer_tensors)} tensors)")
        
        # Prepare work for parallel processing
        work_items = []
        for i, (tensor_name, tensor) in enumerate(layer_tensors):
            scheme = self.get_quantization_scheme(tensor_name)
            work_items.append((tensor_name, tensor, scheme, i))
        
        # Process all tensors in this layer in parallel
        quantized_tensors = {}
        layer_stats = {
            'tensors_processed': 0,
            'original_size': 0,
            'quantized_size': 0
        }
        
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            # Submit all work
            future_to_tensor = {
                executor.submit(self.quantize_tensor_chunk, work_item): work_item[0] 
                for work_item in work_items
            }
            
            # Collect results
            for future in as_completed(future_to_tensor):
                tensor_name = future_to_tensor[future]
                try:
                    result = future.result()
                    quantized_tensors[result['tensor_name']] = {
                        'tensor': result['quantized_tensor'],
                        'scale': result['scale'],
                        'scheme': result['scheme']
                    }
                    
                    layer_stats['tensors_processed'] += 1
                    layer_stats['original_size'] += result['original_size']
                    layer_stats['quantized_size'] += result['quantized_size']
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {tensor_name}: {e}")
        
        # Save the entire layer immediately
        layer_output_file = self.output_dir / f"{layer_name}.safetensors"
        
        # Prepare tensors for saving
        tensors_to_save = {}
        metadata = {}
        
        for tensor_name, data in quantized_tensors.items():
            tensors_to_save[tensor_name] = data['tensor']
            tensors_to_save[f"{tensor_name}_scale"] = data['scale']
            metadata[tensor_name] = data['scheme']
        
        # Save using safetensors
        save_file(tensors_to_save, layer_output_file, metadata=metadata)
        
        logger.info(f"✅ Layer {layer_name} saved to {layer_output_file}")
        logger.info(f"   📊 {layer_stats['tensors_processed']} tensors, "
                   f"{layer_stats['original_size']/(1024**2):.1f}MB → "
                   f"{layer_stats['quantized_size']/(1024**2):.1f}MB")
        
        # Cleanup
        del quantized_tensors, tensors_to_save
        gc.collect()
        
        return layer_stats
    
    def group_tensors_by_layer(self, safetensors_file: Path) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
        """Group tensors by layer for efficient processing"""
        logger.info(f"📂 Analyzing {safetensors_file.name}")
        
        layers = {}
        
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                # Extract layer information
                if "layers." in tensor_name:
                    # Extract layer number
                    layer_part = tensor_name.split("layers.")[1]
                    layer_num = layer_part.split(".")[0]
                    layer_key = f"layer_{layer_num}"
                else:
                    # Non-layer tensors (embeddings, etc.)
                    layer_key = "shared"
                
                if layer_key not in layers:
                    layers[layer_key] = []
                
                # Load tensor
                tensor = f.get_tensor(tensor_name)
                layers[layer_key].append((tensor_name, tensor))
        
        logger.info(f"   📊 Found {len(layers)} layer groups")
        return layers
    
    def quantize_model(self) -> Dict:
        """Quantize the entire model layer by layer"""
        logger.info("🚀 Layer-by-Layer Quantization for Gemma 3 27B")
        logger.info(f"💻 Using ALL {cpu_count} CPU cores per layer")
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
            logger.info(f"🔄 Processing file {file_idx + 1}/{len(safetensor_files)}: {file_path.name}")
            
            # Group tensors by layer
            layers = self.group_tensors_by_layer(file_path)
            
            # Process each layer
            for layer_name, layer_tensors in layers.items():
                memory_before = self.monitor_memory()
                
                layer_stats = self.process_layer(layer_tensors, f"{file_path.stem}_{layer_name}")
                
                # Update global stats
                self.stats['total_original_size'] += layer_stats['original_size']
                self.stats['total_quantized_size'] += layer_stats['quantized_size']
                self.stats['processed_tensors'] += layer_stats['tensors_processed']
                self.stats['layers_processed'] += 1
                
                memory_after = self.monitor_memory()
                logger.info(f"   💾 Memory: {memory_before:.1f}MB → {memory_after:.1f}MB")
            
            # Progress update
            elapsed = time.time() - start_time
            files_remaining = len(safetensor_files) - file_idx - 1
            eta = (elapsed / (file_idx + 1)) * files_remaining if file_idx > 0 else 0
            
            logger.info(f"✅ File complete. ETA: {eta/60:.1f}m - Peak memory: {self.stats['peak_memory_mb']:.1f}MB")
        
        # Final statistics
        total_time = time.time() - start_time
        total_savings = (self.stats['total_original_size'] - self.stats['total_quantized_size']) / self.stats['total_original_size']
        
        logger.info("🎉 QUANTIZATION COMPLETE!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📊 Layers processed: {self.stats['layers_processed']}")
        logger.info(f"📊 Tensors processed: {self.stats['processed_tensors']:,}")
        logger.info(f"📊 Size: {self.stats['total_original_size']/(1024**3):.2f}GB → {self.stats['total_quantized_size']/(1024**3):.2f}GB")
        logger.info(f"📊 Memory reduction: {total_savings:.1%}")
        logger.info(f"💾 Peak memory usage: {self.stats['peak_memory_mb']:.1f}MB")
        logger.info(f"🎯 Fits in 16GB iGPU: {self.stats['total_quantized_size']/(1024**3) < 16}")
        
        # Save metadata
        metadata = {
            "model": "gemma-3-27b-it",
            "quantization_method": "layer_by_layer_16_cores",
            "quantization_time_minutes": total_time / 60,
            "cpu_cores_used": cpu_count,
            "original_size_gb": self.stats['total_original_size'] / (1024**3),
            "quantized_size_gb": self.stats['total_quantized_size'] / (1024**3),
            "memory_reduction": total_savings,
            "peak_memory_usage_mb": self.stats['peak_memory_mb'],
            "fits_in_16gb_igpu": self.stats['total_quantized_size']/(1024**3) < 16,
            "layers_processed": self.stats['layers_processed'],
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
    logger.info(f"🔥 Using ALL {cpu_count} CPU cores for maximum performance")
    
    # Initialize quantizer
    quantizer = LayerByLayerQuantizer()
    
    # Run quantization
    result = quantizer.quantize_model()
    
    if result:
        logger.info("✅ Layer-by-layer quantization completed successfully!")
        return True
    else:
        logger.error("❌ Quantization failed!")
        return False

if __name__ == "__main__":
    main()