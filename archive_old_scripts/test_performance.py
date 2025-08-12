#!/usr/bin/env python3
"""Test performance of the GPU pipeline"""

import logging
import time
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance():
    """Test token generation performance"""
    logger.info("🚀 Starting performance test...")
    
    try:
        # Initialize pipeline
        pipeline = PureHardwarePipelineFixed()
        model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
        
        if not pipeline.initialize(model_path):
            logger.error("Failed to initialize pipeline")
            return
            
        logger.info("✅ Pipeline initialized successfully")
        
        # Test with different token counts
        test_configs = [
            {"input": [72], "max_tokens": 10, "name": "Small"},
            {"input": [72, 101, 108, 108, 111], "max_tokens": 20, "name": "Medium"},
            {"input": [72, 101, 108, 108, 111], "max_tokens": 50, "name": "Large"}
        ]
        
        for config in test_configs:
            logger.info(f"\n📊 Testing {config['name']} generation...")
            logger.info(f"   Input tokens: {len(config['input'])}")
            logger.info(f"   Max new tokens: {config['max_tokens']}")
            
            # Measure time
            start_time = time.time()
            generated_ids = pipeline.generate_tokens(
                config['input'], 
                max_tokens=config['max_tokens'],
                temperature=1.0,
                top_p=0.9
            )
            elapsed = time.time() - start_time
            
            # Calculate TPS
            tokens_generated = len(generated_ids) - len(config['input'])
            tps = tokens_generated / elapsed if elapsed > 0 else 0
            
            logger.info(f"   ✅ Generated {tokens_generated} tokens in {elapsed:.2f}s")
            logger.info(f"   📈 Performance: {tps:.1f} TPS")
            
            # Check against target
            target_tps = 81  # From documentation
            if tps >= target_tps * 0.8:  # Allow 20% margin
                logger.info(f"   ✅ Meeting performance target ({target_tps} TPS)")
            else:
                logger.warning(f"   ⚠️ Below performance target ({target_tps} TPS)")
        
        # Cleanup
        pipeline.cleanup()
        logger.info("\n✅ Performance test completed!")
        
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_performance()