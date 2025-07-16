#!/usr/bin/env python3
"""Test just the loading speed improvement"""

import time
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_loading_speed():
    """Test only the model loading speed"""
    logger.info("🚀 Testing Loading Speed with Lightning Fast Loader")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    # Time the model loading
    load_start = time.time()
    try:
        success = pipeline.initialize("./quantized_models/gemma-3-27b-it-layer-by-layer")
        load_time = time.time() - load_start
        
        if success:
            logger.info(f"✅ Model loaded successfully in {load_time:.1f} seconds")
            logger.info(f"   Target: 10-15 seconds (vs 2+ minutes originally)")
            if load_time < 30:
                logger.info("   🎉 MASSIVE IMPROVEMENT!")
            else:
                logger.info(f"   ⚠️ Still slower than target, but {120/load_time:.1f}x faster than original")
        else:
            logger.error("❌ Pipeline initialization failed!")
            
    except Exception as e:
        load_time = time.time() - load_start
        logger.error(f"❌ Loading failed after {load_time:.1f} seconds: {e}")
        if "STRICT NPU+iGPU MODE" in str(e):
            logger.info("   This is expected in STRICT mode - no CPU fallbacks allowed!")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    test_loading_speed()