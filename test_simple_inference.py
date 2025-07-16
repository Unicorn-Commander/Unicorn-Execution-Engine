#!/usr/bin/env python3
"""Simple test to isolate inference issues"""

import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_inference():
    """Test basic inference with minimal tokens"""
    logger.info("🚀 Starting simple inference test...")
    
    try:
        # Initialize pipeline
        pipeline = PureHardwarePipelineFixed()
        model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
        pipeline.initialize(model_path)
        logger.info("✅ Pipeline initialized successfully")
        
        # Test with single token
        input_ids = [72]  # Just 'H'
        logger.info(f"📝 Testing with input: {input_ids}")
        
        # Generate just 1 token
        generated_ids = pipeline.generate_tokens(
            input_ids, 
            max_tokens=1,
            temperature=1.0,
            top_p=0.9
        )
        
        logger.info(f"✅ Generated IDs: {generated_ids}")
        logger.info(f"   New tokens: {generated_ids[len(input_ids):]}")
        
        # Cleanup
        pipeline.cleanup()
        logger.info("✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_simple_inference()