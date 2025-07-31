#!/usr/bin/env python3
"""
Quick test of 4B model with fixed Vulkan copy operations.
"""

import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_4b_quick():
    logger.info("🚀 Quick 4B Test with Fixed Vulkan...")
    
    # Use 4B model
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return
    
    # Test embedding lookup directly
    logger.info("🔍 Testing embedding lookup...")
    try:
        input_ids = [1, 2, 3]
        embed_info = pipeline.gpu_buffers.get('shared_language_model.model.embed_tokens.weight')
        if embed_info:
            logger.info(f"✅ Embedding weights found: {embed_info['size_mb']:.1f}MB")
            logger.info(f"   Device: {embed_info.get('device', 'unknown')}")
            
            # Try embedding lookup
            hidden_states = pipeline.vulkan_engine.compute_embedding_lookup_gpu(input_ids, embed_info['buffer_info'])
            logger.info(f"✅ Embedding lookup successful: {hidden_states.shape}")
            
        else:
            logger.error("❌ No embedding weights found in GPU buffers")
    except Exception as e:
        logger.error(f"❌ Embedding lookup failed: {e}")
        import traceback
        traceback.print_exc()
    
    pipeline.cleanup()

if __name__ == "__main__":
    test_4b_quick()