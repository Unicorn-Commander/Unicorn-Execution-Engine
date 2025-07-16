#!/usr/bin/env python3
"""
Debug GPU buffer keys to understand what's actually stored
"""

import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_gpu_buffers():
    """Check what keys are actually stored in GPU buffers"""
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model...")
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    logger.info(f"\n✅ Model loaded with {len(pipeline.layer_weights_gpu)} layers")
    
    # Check GPU buffer keys
    logger.info(f"\n📊 GPU Buffers: {len(pipeline.gpu_buffers)} total")
    
    # Look for layer 0 weights
    logger.info("\n🔍 Looking for layer 0 weights:")
    layer_0_keys = [k for k in pipeline.gpu_buffers.keys() if 'layer' in k and '.0.' in k]
    for key in sorted(layer_0_keys)[:10]:  # Show first 10
        logger.info(f"   - {key}")
    
    # Check exact keys we're looking for
    logger.info("\n🔍 Checking specific keys:")
    test_keys = [
        'language_model.model.layers.0.self_attn.q_proj.weight',
        'language_model.model.layers.0.mlp.gate_proj.weight'
    ]
    
    for key in test_keys:
        if key in pipeline.gpu_buffers:
            logger.info(f"   ✅ Found: {key}")
        else:
            logger.info(f"   ❌ Missing: {key}")
            # Look for similar keys
            similar = [k for k in pipeline.gpu_buffers.keys() if 'layers.0' in k and any(part in k for part in ['q_proj', 'gate_proj'])]
            if similar:
                logger.info(f"      Similar keys: {similar[:3]}")
    
    # Check layer_weights_gpu structure
    logger.info(f"\n📊 Layer weights GPU structure:")
    if 0 in pipeline.layer_weights_gpu:
        layer_0_weights = pipeline.layer_weights_gpu[0]
        logger.info(f"   Layer 0 has {len(layer_0_weights)} weight keys:")
        for key in sorted(layer_0_weights.keys())[:5]:
            logger.info(f"   - {key}")
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    debug_gpu_buffers()