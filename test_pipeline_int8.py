#!/usr/bin/env python3
"""Test pipeline with INT8 support and measure memory/performance"""

import time
import logging
import subprocess
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_memory_stats():
    """Get current memory usage"""
    try:
        # Get VRAM
        vram = int(open('/sys/class/drm/card0/device/mem_info_vram_used').read().strip())
        # Get GTT
        gtt = int(open('/sys/class/drm/card0/device/mem_info_gtt_used').read().strip())
        # Get system RAM
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    ram_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                    break
        
        return {
            'vram_mb': vram / (1024 * 1024),
            'gtt_mb': gtt / (1024 * 1024),
            'ram_available_gb': ram_available / (1024 * 1024 * 1024)
        }
    except:
        return None

def main():
    """Test INT8 pipeline"""
    logger.info("🚀 Testing Pipeline with INT8 Support")
    
    # Get baseline memory
    baseline = get_memory_stats()
    if baseline:
        logger.info(f"📊 Baseline Memory:")
        logger.info(f"   VRAM: {baseline['vram_mb']:.1f}MB")
        logger.info(f"   GTT: {baseline['gtt_mb']:.1f}MB")
        logger.info(f"   RAM Available: {baseline['ram_available_gb']:.1f}GB")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("🔄 Loading model with INT8 support...")
    start_time = time.time()
    
    # Load just first 5 layers for quick test
    if pipeline.initialize(model_path):
        load_time = time.time() - start_time
        
        # Get memory after loading
        after = get_memory_stats()
        if after and baseline:
            vram_used = after['vram_mb'] - baseline['vram_mb']
            gtt_used = after['gtt_mb'] - baseline['gtt_mb']
            ram_used = baseline['ram_available_gb'] - after['ram_available_gb']
            
            logger.info(f"✅ Model loaded in {load_time:.1f}s")
            logger.info(f"📊 Memory Usage:")
            logger.info(f"   VRAM: +{vram_used/1024:.1f}GB ({after['vram_mb']/1024:.1f}GB total)")
            logger.info(f"   GTT:  +{gtt_used/1024:.1f}GB ({after['gtt_mb']/1024:.1f}GB total)")
            logger.info(f"   RAM:  +{ram_used:.1f}GB used")
            logger.info(f"   Total GPU: {(vram_used + gtt_used)/1024:.1f}GB")
            
            # Check if INT8 is working (should be ~26GB, not 104GB)
            total_gpu_gb = (vram_used + gtt_used) / 1024
            if total_gpu_gb < 50:  # Less than 50GB means INT8 is working
                logger.info("✅ INT8 weights confirmed! Memory usage is correct.")
            else:
                logger.warning("⚠️ Memory usage too high - possible FP32 conversion!")
        
        # Quick inference test
        logger.info("\n🧪 Testing inference...")
        test_input = "Hello, world!"
        
        try:
            start = time.time()
            # Just test embedding lookup
            input_ids = [1, 2, 3]  # Dummy tokens
            
            embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
            if embed_weight is not None:
                logger.info(f"✅ Embedding weights shape: {embed_weight.shape}")
                
                # Test one layer forward
                import numpy as np
                hidden_states = embed_weight[input_ids]
                if hidden_states.ndim == 2:
                    hidden_states = hidden_states[np.newaxis, :]
                
                output, _ = pipeline.forward_layer(0, hidden_states)
                logger.info(f"✅ Layer 0 forward pass: {output.shape}")
                
                inference_time = time.time() - start
                logger.info(f"⏱️ Inference test time: {inference_time*1000:.1f}ms")
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")
        
        # Cleanup
        pipeline.cleanup()
        
        # Final memory check
        final = get_memory_stats()
        if final:
            logger.info(f"\n📊 Final Memory (after cleanup):")
            logger.info(f"   VRAM: {final['vram_mb']:.1f}MB")
            logger.info(f"   GTT: {final['gtt_mb']:.1f}MB")
    else:
        logger.error("❌ Pipeline initialization failed")

if __name__ == "__main__":
    main()