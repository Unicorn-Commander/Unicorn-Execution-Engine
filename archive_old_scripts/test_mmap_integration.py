#!/usr/bin/env python3
"""
Test the mmap integration with the complete pipeline
"""

import torch
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mmap_integration():
    """Test the complete pipeline with mmap optimization"""
    logger.info("🧪 Testing mmap integration with complete pipeline")
    
    try:
        # Import the pipeline
        from complete_npu_igpu_inference_pipeline import CompleteNPUIGPUInferencePipeline
        
        # Initialize with mmap enabled
        pipeline = CompleteNPUIGPUInferencePipeline(use_fp16=True, use_mmap=True)
        
        # Initialize hardware
        if not pipeline.initialize_hardware():
            logger.error("❌ Hardware initialization failed")
            return False
        
        # Test loading a single layer
        logger.info("🔬 Testing single layer processing...")
        
        # Load layer 0
        layer_weights = pipeline.layer_loader(0)
        logger.info(f"✅ Layer 0 loaded: {len(layer_weights)} weights")
        
        # Create sample input
        input_ids = torch.tensor([[1, 450, 3437]], dtype=torch.long)
        
        # Get embedding
        embed_weight_key = 'language_model.model.embed_tokens.weight'
        embed_weight_info = pipeline.shared_weights[embed_weight_key]
        embed_weight = pipeline._ensure_float_tensor(embed_weight_info)
        hidden_states = torch.nn.functional.embedding(input_ids, embed_weight)
        
        logger.info(f"Hidden states shape: {hidden_states.shape}")
        
        # Test transformer layer computation
        logger.info("🔬 Testing transformer layer computation...")
        output = pipeline.compute_transformer_layer(hidden_states, layer_weights)
        logger.info(f"✅ Transformer layer output shape: {output.shape}")
        
        logger.info("🎉 Mmap integration test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mmap integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_mmap_integration()
    if success:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed!")