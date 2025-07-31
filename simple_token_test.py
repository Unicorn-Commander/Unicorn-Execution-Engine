#!/usr/bin/env python3
"""
Simple test to generate just 1 token without FFN to verify basic NPU+iGPU functionality.
"""

# Fix Python 3.11 compatibility with vulkan
import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test():
    """Test basic token generation with NPU+iGPU."""
    logger.info("🚀 Starting SIMPLE NPU+iGPU test (bypass FFN)...")
    
    # Initialize the pipeline with 4B model
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize the pipeline.")
        return
    
    # Simple input - just generate 1 token
    input_ids = [1, 2, 3]
    logger.info("🔥 Generating 1 token...")
    
    start_time = time.time()
    try:
        # Try to generate just 1 token
        result = pipeline.generate_tokens(input_ids, max_tokens=1)
        end_time = time.time()
        
        logger.info(f"✅ SUCCESS! Generated token in {end_time - start_time:.3f}s")
        logger.info(f"📊 Result: {result}")
        
        # Calculate TPS if successful
        if len(result) > len(input_ids):
            tokens_generated = len(result) - len(input_ids)
            tps = tokens_generated / (end_time - start_time)
            logger.info(f"🚀 BREAKTHROUGH TPS: {tps:.2f} tokens/second")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    simple_test()