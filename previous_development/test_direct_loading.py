#!/usr/bin/env python3
"""
Test direct loading to identify the exact issue
"""

import sys
import logging

logging.basicConfig(level=logging.DEBUG)

# Add more detailed error handling
import pure_hardware_pipeline_final

def test_direct():
    """Test direct pipeline loading"""
    
    print("Testing pipeline initialization...")
    
    pipeline = pure_hardware_pipeline_final.PureHardwarePipelineFinal()
    
    model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
    
    try:
        result = pipeline.initialize(model_path)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct()