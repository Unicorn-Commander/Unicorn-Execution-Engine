#!/usr/bin/env python3
"""Debug script to check layer weight names"""

import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)

def debug_layer_keys():
    """Check what keys are stored in layer weights"""
    
    print("🔍 Debugging Layer Weight Keys")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    if pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer'):
        print("\n✅ Pipeline initialized!")
        
        # Check layer 0 weights
        if 0 in pipeline.layer_weights_gpu:
            print("\n📦 Layer 0 GPU weights:")
            for weight_name, buffer_key in pipeline.layer_weights_gpu[0].items():
                print(f"  Weight: {weight_name}")
                print(f"  Buffer: {buffer_key}")
                print()
        
        # Also check what's in shared weights
        print("\n📦 Sample shared weights (first 5):")
        for i, (key, val) in enumerate(pipeline.shared_weights.items()):
            if i >= 5:
                break
            print(f"  - {key}")
        
        pipeline.cleanup()
    else:
        print("\n❌ Failed to initialize pipeline")


if __name__ == "__main__":
    debug_layer_keys()