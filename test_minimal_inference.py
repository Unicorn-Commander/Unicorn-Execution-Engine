#!/usr/bin/env python3
"""Minimal inference test - single token generation"""

import logging
import time
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_inference():
    """Test minimal inference - just one forward pass"""
    logger.info("🚀 Testing minimal inference...")
    
    try:
        # Initialize pipeline
        pipeline = PureHardwarePipelineFixed()
        model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
        
        if not pipeline.initialize(model_path):
            logger.error("❌ Pipeline initialization failed")
            return False
            
        logger.info("✅ Pipeline initialized")
        logger.info(f"   NPU: {'READY' if pipeline.npu_kernel else 'FALLBACK TO GPU'}")
        
        # Test with minimal input
        input_ids = [72]  # Just 'H'
        logger.info(f"📝 Testing with input: {input_ids}")
        
        # Get embedding directly
        embed_weight = pipeline.get_weight_from_gpu('shared_language_model.model.embed_tokens.weight')
        if embed_weight is None:
            logger.error("❌ Failed to get embedding weights")
            return False
            
        logger.info(f"✅ Embedding weights loaded: {embed_weight.shape}")
        
        # Create initial hidden states
        hidden_states = embed_weight[input_ids]
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[np.newaxis, :]
            
        logger.info(f"✅ Initial hidden states: {hidden_states.shape}")
        
        # Test single layer forward pass
        logger.info("🔄 Testing single layer forward pass...")
        start_time = time.time()
        
        output, kv_cache = pipeline.forward_layer(0, hidden_states)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Layer 0 forward pass completed in {elapsed*1000:.1f}ms")
        logger.info(f"   Output shape: {output.shape}")
        
        # Test attention separately
        if pipeline.npu_kernel:
            logger.info("🧠 NPU available for attention")
        else:
            logger.info("⚡ Using GPU for attention")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()

if __name__ == "__main__":
    success = test_minimal_inference()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")