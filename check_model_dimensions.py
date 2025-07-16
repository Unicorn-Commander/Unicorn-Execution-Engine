#!/usr/bin/env python3
"""
Check actual model dimensions
"""

import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dimensions():
    """Check actual model dimensions"""
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("Loading model...")
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    
    # Check embedding dimension
    embed_key = 'shared_language_model.model.embed_tokens.weight'
    if embed_key in pipeline.gpu_buffers:
        embed_shape = pipeline.gpu_buffers[embed_key].get('shape')
        logger.info(f"Embedding shape: {embed_shape}")
        vocab_size, hidden_dim = embed_shape
        logger.info(f"Hidden dimension: {hidden_dim}")
    else:
        logger.info("Embedding not found in GPU buffers")
    
    # Check layer 0 dimensions
    q_key = 'layer_0_language_model.model.layers.0.self_attn.q_proj.weight'
    if q_key in pipeline.gpu_buffers:
        q_shape = pipeline.gpu_buffers[q_key].get('shape')
        logger.info(f"Q projection shape: {q_shape}")
        logger.info(f"Q heads: 32, head dim: {q_shape[0] // 32}")
    
    # Check FFN dimensions
    gate_key = 'layer_0_language_model.model.layers.0.mlp.gate_proj.weight'
    if gate_key in pipeline.gpu_buffers:
        gate_shape = pipeline.gpu_buffers[gate_key].get('shape')
        logger.info(f"Gate projection shape: {gate_shape}")
        logger.info(f"FFN intermediate size: {gate_shape[0]}")
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    check_dimensions()