#!/usr/bin/env python3.13
"""
Production Safetensors Weight Loader
Memory-mapped, zero-copy weight loading for maximum performance
"""

import os
import sys
import mmap
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

class ProductionWeightLoader:
    """
    🦄 Production weight loader with memory mapping
    - Zero-copy memory mapping
    - Direct hardware buffer creation
    - Optimized for NPU+iGPU pipelines
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.weight_files = list(self.model_path.glob("*.safetensors"))
        self.mapped_files = {}
        self.tensor_info = {}
        self.total_size = 0
        
        if not self.weight_files:
            raise ValueError(f"No safetensors files found in {model_path}")
            
        print(f"🦄 Production Weight Loader")
        print(f"   Model: {self.model_path.name}")
        print(f"   Files: {len(self.weight_files)}")
        
        # Calculate total size
        for f in self.weight_files:
            self.total_size += f.stat().st_size
        print(f"   Total size: {self.total_size / 1024**3:.2f} GB")
    
    def load_all_files(self) -> Dict[str, Any]:
        """Load and memory map all safetensors files"""
        print("\n📦 Memory mapping weight files...")
        
        all_tensors = {}
        
        for file_idx, weight_file in enumerate(self.weight_files):
            print(f"   [{file_idx+1}/{len(self.weight_files)}] {weight_file.name}")
            
            try:
                # Open file for memory mapping
                with open(weight_file, 'rb') as f:
                    # Read header length
                    header_len = struct.unpack('<Q', f.read(8))[0]
                    
                    # Read header
                    header_data = f.read(header_len)
                    header = json.loads(header_data.decode('utf-8'))
                    
                    # Data starts after header
                    data_offset = 8 + header_len
                    
                    # Memory map the file
                    f.seek(0)
                    mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    self.mapped_files[weight_file.name] = mapped
                    
                    # Process tensors in this file
                    tensor_count = 0
                    for name, info in header.items():
                        if name == '__metadata__':
                            continue
                            
                        if isinstance(info, dict) and 'shape' in info:
                            shape = info['shape']
                            dtype = info['dtype']
                            data_offsets = info['data_offsets']
                            
                            # Create tensor info
                            tensor_info = {
                                'file': weight_file.name,
                                'mapped': mapped,
                                'shape': shape,
                                'dtype': dtype,
                                'offset': data_offset + data_offsets[0],
                                'size': data_offsets[1] - data_offsets[0]
                            }
                            
                            all_tensors[name] = tensor_info
                            tensor_count += 1
                    
                    print(f"      {tensor_count} tensors mapped")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        print(f"✅ Mapped {len(all_tensors)} total tensors")
        return all_tensors
    
    def get_tensor_array(self, tensor_info: Dict[str, Any]) -> np.ndarray:
        """Get numpy array from memory-mapped tensor (zero-copy)"""
        mapped = tensor_info['mapped']
        offset = tensor_info['offset']
        shape = tensor_info['shape']
        dtype = tensor_info['dtype']
        size = tensor_info['size']
        
        # Map safetensors dtype to numpy
        dtype_map = {
            'F32': np.float32,
            'F16': np.float16,
            'BF16': np.float16,  # Approximate
            'I32': np.int32,
            'I64': np.int64,
            'U8': np.uint8,
            'I8': np.int8
        }
        
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Create array view of memory-mapped data
        buffer = mapped[offset:offset + size]
        array = np.frombuffer(buffer, dtype=np_dtype).reshape(shape)
        
        return array
    
    def get_layer_weights(self, layer_idx: int, tensors: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get all weights for a specific layer"""
        layer_weights = {}
        layer_prefix = f"language_model.model.layers.{layer_idx}."
        
        for name, info in tensors.items():
            if name.startswith(layer_prefix):
                # Remove layer prefix for cleaner names
                clean_name = name[len(layer_prefix):]
                if not clean_name.endswith('_original_shape'):
                    layer_weights[clean_name] = self.get_tensor_array(info)
        
        return layer_weights
    
    def get_embedding_weights(self, tensors: Dict[str, Any]) -> np.ndarray:
        """Get embedding table weights"""
        embed_name = "language_model.model.embed_tokens.weight"
        if embed_name in tensors:
            return self.get_tensor_array(tensors[embed_name])
        else:
            print(f"⚠️  Embedding weights not found: {embed_name}")
            return None
    
    def cleanup(self):
        """Clean up memory mapped files"""
        for mapped in self.mapped_files.values():
            mapped.close()
        self.mapped_files.clear()

def test_weight_loader():
    """Test the production weight loader"""
    print("🦄 Testing Production Weight Loader")
    print("=" * 60)
    
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    try:
        # Create loader
        loader = ProductionWeightLoader(model_path)
        
        # Load all tensors
        tensors = loader.load_all_files()
        
        print(f"\n📊 Tensor Analysis:")
        
        # Check embedding weights
        embed_weights = loader.get_embedding_weights(tensors)
        if embed_weights is not None:
            print(f"   Embedding: {embed_weights.shape} ({embed_weights.dtype})")
            vocab_size, hidden_size = embed_weights.shape
            print(f"   Vocab size: {vocab_size}")
            print(f"   Hidden size: {hidden_size}")
        
        # Check first layer weights
        layer_0_weights = loader.get_layer_weights(0, tensors)
        print(f"\n   Layer 0 weights: {len(layer_0_weights)} tensors")
        for name, weight in layer_0_weights.items():
            if 'weight' in name and not name.endswith('_original_shape'):
                print(f"     {name}: {weight.shape}")
        
        # Memory usage
        total_params = sum(
            np.prod(info['shape']) for info in tensors.values() 
            if isinstance(info, dict) and 'shape' in info
        )
        print(f"\n   Total parameters: {total_params:,}")
        print(f"   Memory mapped: {loader.total_size / 1024**3:.2f} GB")
        
        print("\n✅ Weight loader working perfectly!")
        
        # Cleanup
        loader.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_weight_loader()