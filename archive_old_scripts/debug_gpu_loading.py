#!/usr/bin/env python3
"""Debug script to trace GPU loading issues"""

import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_loading():
    """Debug the GPU loading process"""
    pipeline = PureHardwarePipelineFixed()
    
    # Initialize the loader
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    pipeline.loader = pipeline._initialize_loader(model_path)
    
    # Load model info
    model_info = pipeline.loader.load_model()
    pipeline.shared_weights = model_info.get('shared_weights', {})
    pipeline.layer_loader = model_info.get('layer_loader')
    
    # Debug shared weights
    logger.info(f"Shared weights keys: {list(pipeline.shared_weights.keys())[:5]}...")
    
    # Test loading a single layer
    logger.info("Testing layer loader...")
    if pipeline.layer_loader:
        layer_0_weights = pipeline.layer_loader(0)
        logger.info(f"Layer 0 weight keys: {list(layer_0_weights.keys())[:5]}...")
        
        # Check the structure
        for key, value in list(layer_0_weights.items())[:2]:
            logger.info(f"Weight '{key}' type: {type(value)}")
            if isinstance(value, dict):
                logger.info(f"  Dict keys: {list(value.keys())}")
    else:
        logger.error("No layer_loader found!")

if __name__ == "__main__":
    debug_loading()