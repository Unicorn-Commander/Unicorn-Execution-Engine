#!/usr/bin/env python3
"""Simple test for model loading with GPU allocation"""

import time
import subprocess
from pure_hardware_pipeline import PureHardwarePipeline

def get_gpu_memory():
    """Get current GPU memory usage"""
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    return result.stdout

print("Starting model load test...")
print("Monitor GPU memory in another terminal with: watch -n 0.5 'radeontop -d -'")

# Get baseline
print("\nBaseline GPU memory:")
print(get_gpu_memory())

# Initialize pipeline with timeout
pipeline = PureHardwarePipeline()

import signal
def timeout_handler(signum, frame):
    print("\nTimeout! Current GPU memory:")
    print(get_gpu_memory())
    raise TimeoutError("Model loading timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(120)  # 2 minute timeout

try:
    print("\nInitializing model...")
    pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer')
    signal.alarm(0)
    
    print("\n✅ Model loaded! Final GPU memory:")
    print(get_gpu_memory())
    
    # Test generation
    print("\nTesting generation...")
    result = pipeline.generate_tokens([1, 2, 3], max_tokens=5)
    print(f"Generated: {result}")
    
except TimeoutError:
    print("\n❌ Model loading is using CPU RAM instead of GPU")
except Exception as e:
    print(f"\n❌ Error: {e}")
finally:
    signal.alarm(0)