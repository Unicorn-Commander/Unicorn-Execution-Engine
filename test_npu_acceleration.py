#!/usr/bin/env python3
"""Test NPU acceleration"""

import numpy as np
import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_npu():
    logger.info("🚀 Testing NPU Acceleration...")
    
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize")
        return
    
    # Check if NPU is available
    if pipeline.npu_kernel:
        logger.info("✅ NPU kernel available!")
        logger.info(f"   Type: {type(pipeline.npu_kernel).__name__}")
    else:
        logger.error("❌ No NPU kernel available")
        return
    
    # Test a simple forward pass
    try:
        logger.info("🔄 Testing single token generation...")
        start = time.time()
        result = pipeline.generate_tokens([1, 2, 3], max_tokens=1)
        elapsed = time.time() - start
        
        logger.info(f"✅ Generated: {result}")
        logger.info(f"⏱️ Time: {elapsed:.2f}s")
        
        if pipeline.npu_total_layers > 0:
            npu_avg = (pipeline.npu_total_time / pipeline.npu_total_layers) * 1000
            logger.info(f"🧠 NPU Average: {npu_avg:.2f}ms per layer")
            logger.info(f"🚀 NPU Layers: {pipeline.npu_total_layers}")
        else:
            logger.warning("⚠️ No NPU layers executed")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    pipeline.cleanup()

if __name__ == "__main__":
    test_npu()