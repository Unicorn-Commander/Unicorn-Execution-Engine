#!/usr/bin/env python3
"""
Quick test to verify performance and GPU usage
"""

import logging
import time
import subprocess
import numpy as np
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Check current GPU memory usage"""
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'vram' in line and 'gtt' in line:
            return line
    return "GPU info not found"

def main():
    logger.info("🚀 Quick Performance Test")
    
    # Check initial GPU state
    logger.info(f"Initial GPU: {check_gpu_memory()}")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineGPUFixed()
    model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    
    # Time model loading
    logger.info("Loading model...")
    start_load = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Check GPU memory after loading
    logger.info(f"After loading GPU: {check_gpu_memory()}")
    
    # Test inference performance
    logger.info("\n🧪 Testing inference performance...")
    test_input = np.random.randn(1, 1, 5376).astype(np.float32)
    
    # Warm up
    pipeline.forward_layer(0, test_input)
    
    # Time 10 layers
    start_inf = time.time()
    for i in range(10):
        output, _ = pipeline.forward_layer(i, test_input)
    
    inf_time = time.time() - start_inf
    avg_layer_time = inf_time / 10 * 1000  # ms
    
    logger.info(f"✅ Average layer time: {avg_layer_time:.1f}ms")
    logger.info(f"📊 Estimated TPS: {1000/avg_layer_time:.1f}")
    
    # Check GPU during inference
    logger.info(f"During inference GPU: {check_gpu_memory()}")
    
    # Test Magic Unicorn prompt
    logger.info("\n🦄 Testing Magic Unicorn prompt...")
    prompt = "Magic Unicorn Unconventional Technology & Stuff is"
    
    # Simple tokenization
    input_ids = [ord(c) % 1000 for c in prompt]
    
    try:
        start_gen = time.time()
        generated_ids = pipeline.generate_tokens(input_ids, max_tokens=10)
        gen_time = time.time() - start_gen
        
        logger.info(f"✅ Generated {len(generated_ids)} tokens in {gen_time:.1f}s")
        logger.info(f"⚡ TPS: {len(generated_ids)/gen_time:.1f}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    
    logger.info("\n✅ Test complete!")

if __name__ == "__main__":
    main()