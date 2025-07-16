#!/usr/bin/env python3
"""Debug script to check GPU buffer keys"""

import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO)

def debug_buffers():
    """Check what keys are stored in GPU buffers"""
    
    print("🔍 Debugging GPU Buffer Keys")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    
    if pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer'):
        print("\n✅ Pipeline initialized!")
        
        # Print all GPU buffer keys
        print("\n📦 GPU Buffer Keys (shared):")
        for key in sorted(pipeline.gpu_buffers.keys()):
            if key.startswith('shared_'):
                print(f"  - {key}")
        
        # Check shared weights
        print("\n📦 Shared Weights Keys:")
        for key in sorted(pipeline.shared_weights.keys()):
            if 'embed' in key or 'norm' in key:
                print(f"  - {key}")
        
        pipeline.cleanup()
    else:
        print("\n❌ Failed to initialize pipeline")


if __name__ == "__main__":
    debug_buffers()