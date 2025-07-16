#!/usr/bin/env python3
"""Test GPU pipeline with memory monitoring"""

import time
import subprocess
import logging
from pure_hardware_pipeline_optimized import PureHardwarePipelineOptimized

logging.basicConfig(level=logging.INFO)

def get_gpu_memory():
    """Get current GPU memory usage"""
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    return result.stdout

print("🚀 Testing GPU Pipeline with Memory Monitoring")
print("=" * 60)

# Get baseline
print("\n📊 Baseline GPU Memory:")
print(get_gpu_memory())

# Initialize pipeline
print("\n🔄 Initializing pipeline...")
pipeline = PureHardwarePipelineOptimized()

try:
    if pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer'):
        print("\n✅ Pipeline initialized!")
        
        # Check GPU memory after initialization
        print("\n📊 GPU Memory After Initialization:")
        print(get_gpu_memory())
        
        # Keep process alive for monitoring
        print("\n⏳ Keeping process alive for 30 seconds...")
        print("Monitor GPU in another terminal with: watch -n 0.5 'radeontop -d -'")
        
        for i in range(6):
            time.sleep(5)
            print(f"\n📊 GPU Memory at {(i+1)*5}s:")
            print(get_gpu_memory())
        
        # Test generation
        print("\n🔥 Testing token generation...")
        tokens = pipeline.generate_tokens([1, 2, 3], max_tokens=10)
        print(f"Generated tokens: {tokens}")
        
        # Final check
        print("\n📊 Final GPU Memory:")
        print(get_gpu_memory())
        
    else:
        print("\n❌ Failed to initialize pipeline")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'pipeline' in locals():
        pipeline.cleanup()
    print("\n✅ Test complete")