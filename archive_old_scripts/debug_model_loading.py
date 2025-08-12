#!/usr/bin/env python3
"""
Debug Model Loading - Find out why model isn't loading to VRAM
"""

import logging
import subprocess
import time
import gc

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_memory():
    """Check current memory usage"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=1)
        if result.stdout:
            import re
            vram_match = re.search(r'vram\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            gtt_match = re.search(r'gtt\s+(\d+\.\d+)%\s+(\d+\.\d+)mb', result.stdout)
            
            if vram_match and gtt_match:
                vram_mb = float(vram_match.group(2))
                gtt_mb = float(gtt_match.group(2))
                return vram_mb, gtt_mb
    except:
        pass
    return 0, 0

def main():
    logger.info("🔍 DEBUG: Model Loading to VRAM")
    
    # Check initial memory
    vram_start, gtt_start = check_memory()
    logger.info(f"Initial VRAM: {vram_start:.0f}MB, GTT: {gtt_start:.0f}MB")
    
    # Import modules
    logger.info("Importing modules...")
    from pure_mmap_loader import PureMemoryMappedLoader
    from real_vulkan_matrix_compute import VulkanMatrixCompute
    import numpy as np
    
    # Initialize Vulkan
    logger.info("Initializing Vulkan...")
    vulkan = VulkanMatrixCompute()
    vulkan.initialize()
    
    vram_after_init, gtt_after_init = check_memory()
    logger.info(f"After Vulkan init - VRAM: {vram_after_init:.0f}MB (+{vram_after_init-vram_start:.0f}MB)")
    
    # Initialize loader
    logger.info("Initializing model loader...")
    loader = PureMemoryMappedLoader("quantized_models/gemma-3-27b-it-layer-by-layer")
    
    # Load just shared weights first
    logger.info("Loading shared weights...")
    shared_weights = loader.load_shared_weights()
    
    vram_after_shared, gtt_after_shared = check_memory()
    logger.info(f"After shared weights - VRAM: {vram_after_shared:.0f}MB (+{vram_after_shared-vram_after_init:.0f}MB)")
    
    # Try to allocate some weights to VRAM
    logger.info("\nTesting VRAM allocation...")
    
    # Get embeddings (usually one of the largest shared weights)
    embed_key = 'language_model.model.embed_tokens.weight'
    if embed_key in shared_weights:
        embeddings = shared_weights[embed_key]
        logger.info(f"Found embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}, size={embeddings.nbytes/(1024**2):.1f}MB")
        
        # Try to allocate to GPU
        try:
            logger.info("Allocating embeddings to VRAM...")
            gpu_buffer = vulkan._allocate_gpu_memory(embeddings)
            
            # Force sync
            time.sleep(1)
            
            vram_after_embed, gtt_after_embed = check_memory()
            logger.info(f"After embedding allocation - VRAM: {vram_after_embed:.0f}MB (+{vram_after_embed-vram_after_shared:.0f}MB)")
            
            if vram_after_embed > vram_after_shared + 100:  # At least 100MB increase
                logger.info("✅ VRAM allocation working!")
            else:
                logger.warning("⚠️ VRAM didn't increase as expected")
                
        except Exception as e:
            logger.error(f"❌ Failed to allocate embeddings: {e}")
    
    # Load and allocate layers
    logger.info("\nLoading model layers...")
    
    total_vram_target = 0
    total_gtt_target = 0
    
    for layer_idx in range(5):  # Just test first 5 layers
        try:
            logger.info(f"\nLoading layer {layer_idx}...")
            layer_data = loader.load_layer(layer_idx)
            
            if layer_data:
                layer_size = sum(t.nbytes for t in layer_data.values()) / (1024**2)
                logger.info(f"Layer {layer_idx} size: {layer_size:.1f}MB")
                
                # Allocate to VRAM if layer < 20, else GTT
                if layer_idx < 20:
                    logger.info(f"Allocating layer {layer_idx} to VRAM...")
                    for name, tensor in layer_data.items():
                        try:
                            gpu_buffer = vulkan._allocate_gpu_memory(tensor)
                            total_vram_target += tensor.nbytes / (1024**2)
                        except Exception as e:
                            logger.error(f"Failed to allocate {name}: {e}")
                else:
                    logger.info(f"Layer {layer_idx} would go to GTT")
                    total_gtt_target += layer_size
                
                # Check memory after each layer
                vram_current, gtt_current = check_memory()
                logger.info(f"Current VRAM: {vram_current:.0f}MB (target: {total_vram_target:.0f}MB)")
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to load layer {layer_idx}: {e}")
    
    # Final memory check
    time.sleep(2)
    vram_final, gtt_final = check_memory()
    
    logger.info("\n📊 FINAL MEMORY USAGE:")
    logger.info(f"VRAM: {vram_final:.0f}MB (started: {vram_start:.0f}MB, increase: {vram_final-vram_start:.0f}MB)")
    logger.info(f"GTT: {gtt_final:.0f}MB (started: {gtt_start:.0f}MB, increase: {gtt_final-gtt_start:.0f}MB)")
    logger.info(f"Target was: {total_vram_target:.0f}MB VRAM")
    
    # Check Vulkan's internal tracking
    gpu_usage = vulkan.get_memory_usage()
    logger.info(f"\nVulkan internal tracking:")
    logger.info(f"Allocated buffers: {gpu_usage['allocated_buffers']}")
    logger.info(f"Memory usage: {gpu_usage['memory_usage_mb']:.1f}MB")

if __name__ == "__main__":
    main()