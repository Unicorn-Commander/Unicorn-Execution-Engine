#!/usr/bin/env python3
"""Test direct GPU loading without CPU memory"""

import numpy as np
import time
from real_vulkan_matrix_compute import VulkanMatrixCompute
from pure_mmap_loader import PureMemoryMappedLoader

def test_direct_gpu_loading():
    print("Testing direct GPU loading approach...")
    
    # Initialize components
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        print("Failed to initialize Vulkan")
        return
    
    from pure_mmap_loader import MemoryMappedOptimizedLoader
    loader = MemoryMappedOptimizedLoader('quantized_models/gemma-3-27b-it-layer-by-layer')
    
    # Load model info
    print("\nLoading model structure...")
    model_info = loader.load_model()
    layer_loader = model_info.get('layer_loader')
    
    print("\nTesting direct GPU allocation for first few layers...")
    
    # Track memory
    import subprocess
    def get_memory():
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True)
        return result.stdout
    
    print("\nBaseline GPU memory:")
    print(get_memory())
    
    # Try loading first 3 layers directly to GPU
    for layer_idx in range(3):
        print(f"\nLoading layer {layer_idx}...")
        layer_weights = layer_loader(layer_idx)
        
        layer_size_mb = 0
        gpu_buffers = []
        
        for weight_name, weight_info in layer_weights.items():
            if weight_name.startswith('language_model'):
                try:
                    # Get tensor shape and dtype without loading to CPU
                    shape = weight_info.get('shape', [])
                    dtype = weight_info.get('dtype', 'float32')
                    
                    # Calculate size
                    elements = 1
                    for dim in shape:
                        elements *= dim
                    
                    if dtype == 'int8':
                        bytes_per_element = 1
                    elif dtype == 'int4':
                        bytes_per_element = 0.5
                    else:
                        bytes_per_element = 4
                    
                    size_bytes = int(elements * bytes_per_element)
                    size_mb = size_bytes / (1024 * 1024)
                    
                    print(f"  {weight_name}: {shape} {dtype} = {size_mb:.1f}MB")
                    
                    # Create dummy tensor for now (in real implementation, would mmap directly to GPU)
                    dummy_tensor = np.zeros((size_bytes,), dtype=np.uint8)
                    
                    # Allocate to GPU
                    gpu_buffer = vulkan._allocate_gpu_memory(dummy_tensor)
                    gpu_buffers.append(gpu_buffer)
                    
                    layer_size_mb += size_mb
                    
                except Exception as e:
                    print(f"  Error with {weight_name}: {e}")
        
        print(f"Layer {layer_idx} total: {layer_size_mb:.1f}MB")
    
    print("\nFinal GPU memory:")
    print(get_memory())
    
    # Keep buffers alive
    time.sleep(2)
    
    print("\nDirect GPU loading test complete!")
    print("Key insight: Need to allocate GPU memory BEFORE loading tensor data")
    print("Current approach loads to CPU first, then copies to GPU")

if __name__ == "__main__":
    test_direct_gpu_loading()