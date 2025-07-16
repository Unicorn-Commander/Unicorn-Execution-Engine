#!/usr/bin/env python3
"""
Test script for layer-by-layer quantization
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_layer_quantization():
    """Test the layer-by-layer quantization approach"""
    
    # Check if model exists
    model_path = Path("./models/gemma-3-27b-it")
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        return False
    
    # Check safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        logger.error("❌ No safetensors files found!")
        return False
    
    logger.info(f"✅ Found {len(safetensor_files)} safetensors files")
    
    # Test import
    try:
        from layer_by_layer_quantize import LayerByLayerQuantizer
        logger.info("✅ Successfully imported LayerByLayerQuantizer")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    
    # Initialize quantizer
    try:
        quantizer = LayerByLayerQuantizer()
        logger.info("✅ Successfully initialized quantizer")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False
    
    # Test tensor grouping on first file
    try:
        first_file = safetensor_files[0]
        logger.info(f"🔍 Testing tensor grouping on {first_file.name}")
        
        layers = quantizer.group_tensors_by_layer(first_file)
        
        logger.info(f"✅ Found {len(layers)} layer groups:")
        for layer_name, tensors in layers.items():
            logger.info(f"   - {layer_name}: {len(tensors)} tensors")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tensor grouping failed: {e}")
        return False

if __name__ == "__main__":
    # Activate environment
    logger.info("🔧 Testing layer-by-layer quantization approach...")
    
    success = test_layer_quantization()
    
    if success:
        logger.info("✅ All tests passed! Ready for full quantization.")
        logger.info("💡 Run: python layer_by_layer_quantize.py")
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)