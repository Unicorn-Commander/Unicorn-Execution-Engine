#!/usr/bin/env python3
"""
Quick test just for one layer
"""

import fix_vulkan_imports

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_one_layer():
    logger.info("🚀 Testing one layer only...")
    
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize")
        return
    
    # Test just layer 0
    hidden_states = np.random.randn(1, 3, 2560).astype(np.float32)
    
    try:
        logger.info("🔄 Testing layer 0...")
        output, kv_cache = pipeline.forward_layer(0, hidden_states)
        logger.info(f"✅ Layer 0 output shape: {output.shape}")
    except Exception as e:
        logger.error(f"❌ Layer 0 failed: {e}")
        import traceback
        traceback.print_exc()
    
    pipeline.cleanup()

if __name__ == "__main__":
    test_one_layer()