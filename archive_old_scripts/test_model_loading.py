#!/usr/bin/env python3
"""
Test model loading to debug the issue
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the quantized model"""
    
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info(f"🔍 Testing model loading from: {model_path}")
    
    # Check directory exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        return
    
    # List files
    files = list(Path(model_path).glob("*.safetensors"))
    logger.info(f"Found {len(files)} safetensors files")
    
    # Check for shared weights
    shared_files = [f for f in files if "shared" in f.name]
    layer_files = [f for f in files if "layer" in f.name]
    
    logger.info(f"   Shared weight files: {len(shared_files)}")
    logger.info(f"   Layer files: {len(layer_files)}")
    
    # Check quantization results
    quant_results = Path(model_path) / "quantization_results.json"
    if quant_results.exists():
        with open(quant_results) as f:
            results = json.load(f)
        logger.info(f"   Quantization info available: {list(results.keys())}")
    
    # Try to load a safetensors file header
    if files:
        test_file = files[0]
        logger.info(f"\n🧪 Testing file: {test_file.name}")
        
        # Read first 8 bytes to check format
        with open(test_file, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, 'little')
            logger.info(f"   Header size: {header_size} bytes")
            
            # Read header
            header_bytes = f.read(header_size)
            try:
                header = json.loads(header_bytes)
                logger.info(f"   Header keys: {list(header.keys())[:5]}...")
                
                # Check for metadata
                if '__metadata__' in header:
                    logger.info(f"   Metadata: {header['__metadata__']}")
            except Exception as e:
                logger.error(f"   Failed to parse header: {e}")
    
    # Test with pure_mmap_loader
    try:
        from pure_mmap_loader import MemoryMappedOptimizedLoader
        loader = MemoryMappedOptimizedLoader(model_path)
        
        logger.info("\n🔄 Testing MemoryMappedOptimizedLoader...")
        model_data = loader.load_model()
        
        if model_data:
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Shared weights: {len(model_data.get('shared_weights', {}))}")
            if 'layer_loader' in model_data:
                logger.info(f"   Layer loader: Available")
        
    except Exception as e:
        logger.error(f"❌ Loader failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()