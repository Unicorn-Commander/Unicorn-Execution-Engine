#!/usr/bin/env python3
"""Test the optimized pipeline with Lightning Fast Loader and strict NPU+iGPU mode"""

import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimized_pipeline():
    """Test the optimized pipeline"""
    logger.info("🚀 Testing Optimized Pipeline with Lightning Fast Loader")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    # Time the model loading
    load_start = time.time()
    success = pipeline.initialize("./quantized_models/gemma-3-27b-it-layer-by-layer")
    load_time = time.time() - load_start
    
    if not success:
        logger.error("❌ Pipeline initialization failed!")
        return
        
    logger.info(f"✅ Model loaded in {load_time:.1f} seconds (target: 10-15s)")
    logger.info("=" * 80)
    
    # Test inference
    test_prompt = "The future of AI is"
    logger.info(f"📝 Test prompt: '{test_prompt}'")
    
    inference_start = time.time()
    try:
        # Generate tokens
        output = pipeline.generate_tokens(test_prompt, max_tokens=10)
        inference_time = time.time() - inference_start
        
        # Calculate TPS
        tokens_generated = len(output.split()) - len(test_prompt.split())
        tps = tokens_generated / inference_time if inference_time > 0 else 0
        
        logger.info(f"📊 Results:")
        logger.info(f"   Generated: {output}")
        logger.info(f"   Inference time: {inference_time:.2f}s")
        logger.info(f"   Tokens/second: {tps:.1f} TPS (target: 81+ TPS)")
        logger.info(f"   CPU usage: Should be 0% during inference")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        logger.error(f"   This is expected in STRICT mode if hardware is not available")
    
    logger.info("=" * 80)
    logger.info("🏁 Test complete!")

if __name__ == "__main__":
    test_optimized_pipeline()