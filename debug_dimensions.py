#!/usr/bin/env python3
"""
Debug dimension issues in the pipeline
"""

import logging
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 Debugging dimension issues...")
    
    # Create pipeline
    pipeline = PureHardwarePipelineGPUFixed()
    
    # Check model structure
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    # Just initialize to load metadata
    logger.info("Loading model metadata...")
    from pure_mmap_loader import MemoryMappedOptimizedLoader
    loader = MemoryMappedOptimizedLoader(model_path)
    model_info = loader.load_model()
    
    # Check shared weights
    shared_weights = model_info.get('shared_weights', {})
    
    # Check embedding dimension
    embed_key = 'language_model.model.embed_tokens.weight'
    if embed_key in shared_weights:
        embed_info = shared_weights[embed_key]
        logger.info(f"Embedding shape: {embed_info.get('shape', 'unknown')}")
    
    # Check layer 0 weights
    layer_loader = model_info.get('layer_loader')
    if layer_loader:
        layer_0 = layer_loader(0)
        
        # Check attention weights
        q_key = 'language_model.model.layers.0.self_attn.q_proj.weight'
        k_key = 'language_model.model.layers.0.self_attn.k_proj.weight'
        v_key = 'language_model.model.layers.0.self_attn.v_proj.weight'
        o_key = 'language_model.model.layers.0.self_attn.o_proj.weight'
        
        for key, name in [(q_key, 'Q'), (k_key, 'K'), (v_key, 'V'), (o_key, 'O')]:
            if key in layer_0:
                weight_info = layer_0[key]
                shape = weight_info.get('shape', 'unknown')
                logger.info(f"{name} projection shape: {shape}")
        
        # Check FFN weights
        gate_key = 'language_model.model.layers.0.mlp.gate_proj.weight'
        up_key = 'language_model.model.layers.0.mlp.up_proj.weight'
        down_key = 'language_model.model.layers.0.mlp.down_proj.weight'
        
        for key, name in [(gate_key, 'Gate'), (up_key, 'Up'), (down_key, 'Down')]:
            if key in layer_0:
                weight_info = layer_0[key]
                shape = weight_info.get('shape', 'unknown')
                logger.info(f"{name} projection shape: {shape}")
    
    logger.info("\n📊 Expected dimensions for Gemma 27B:")
    logger.info("Hidden size: 5376")
    logger.info("Num attention heads: 32")
    logger.info("Num KV heads: 16 (GQA)")
    logger.info("Head dim: 168 (5376/32)")
    logger.info("Q projection: [4096, 5376] -> outputs 4096")
    logger.info("K/V projection: [2048, 5376] -> outputs 2048")

if __name__ == "__main__":
    main()