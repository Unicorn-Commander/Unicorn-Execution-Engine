#!/usr/bin/env python3
"""
Minimal test - just try to generate one token
"""

import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def minimal_test():
    logger.info("🚀 Minimal Test - Just 1 Token...")
    
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize")
        return
    
    try:
        logger.info("🔄 Generating 1 token...")
        start_time = time.time()
        result = pipeline.generate_tokens([1, 2, 3], max_tokens=1)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ SUCCESS! Generated: {result}")
        logger.info(f"⏱️ Time: {elapsed:.2f}s")
        logger.info(f"🚀 TPS: {1/elapsed:.2f} tokens/second")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        pipeline.cleanup()
    except:
        logger.warning("⚠️ Cleanup warning (expected)")

if __name__ == "__main__":
    minimal_test()