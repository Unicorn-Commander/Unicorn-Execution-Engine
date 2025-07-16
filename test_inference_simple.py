#!/usr/bin/env python3
"""Simple test of the refactored inference pipeline"""

import logging
import time
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Import the pipeline
from pure_hardware_pipeline import PureHardwarePipeline

def monitor_memory():
    """Quick memory check"""
    # GPU
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    gpu_line = result.stdout.strip().split('\n')[-1] if result.stdout else "No GPU data"
    
    # Extract VRAM and GTT
    vram_mb = gtt_mb = 0
    if 'vram' in gpu_line:
        parts = gpu_line.split(',')
        for part in parts:
            if 'vram' in part and 'mb' in part:
                vram_mb = float(part.strip().split()[-1].replace('mb', ''))
            elif 'gtt' in part and 'mb' in part:
                gtt_mb = float(part.strip().split()[-1].replace('mb', ''))
    
    # System RAM
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
        mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) / 1024 / 1024
        mem_avail = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) / 1024 / 1024
        mem_used = mem_total - mem_avail
    
    return {
        'vram_mb': vram_mb,
        'gtt_mb': gtt_mb,
        'ram_gb': mem_used,
        'ram_total_gb': mem_total
    }

print("🚀 Testing Refactored GPU Inference Pipeline")
print("=" * 60)

# Baseline memory
baseline = monitor_memory()
print(f"\n📊 Baseline Memory:")
print(f"   VRAM: {baseline['vram_mb']:.1f} MB")
print(f"   GTT: {baseline['gtt_mb']:.1f} MB")
print(f"   RAM: {baseline['ram_gb']:.1f} GB / {baseline['ram_total_gb']:.1f} GB")

# Initialize pipeline
print("\n🔄 Initializing pipeline...")
pipeline = PureHardwarePipeline()

# Set a timeout
import signal
signal.alarm(300)  # 5 minute timeout

try:
    # Initialize
    start_time = time.time()
    success = pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer')
    init_time = time.time() - start_time
    
    if not success:
        print("❌ Pipeline initialization failed")
        exit(1)
    
    print(f"\n✅ Pipeline initialized in {init_time:.1f} seconds")
    
    # Check memory after init
    after_init = monitor_memory()
    print(f"\n📊 Memory After Initialization:")
    print(f"   VRAM: {after_init['vram_mb']:.1f} MB (+{after_init['vram_mb']-baseline['vram_mb']:.1f} MB)")
    print(f"   GTT: {after_init['gtt_mb']:.1f} MB (+{after_init['gtt_mb']-baseline['gtt_mb']:.1f} MB)")
    print(f"   RAM: {after_init['ram_gb']:.1f} GB (+{after_init['ram_gb']-baseline['ram_gb']:.1f} GB)")
    
    # Check if GPU memory was allocated
    vram_increase = after_init['vram_mb'] - baseline['vram_mb']
    gtt_increase = after_init['gtt_mb'] - baseline['gtt_mb']
    
    if vram_increase > 1000:  # More than 1GB VRAM increase
        print(f"\n✅ GPU memory allocation detected! VRAM increased by {vram_increase/1024:.1f} GB")
    else:
        print(f"\n⚠️ Low VRAM increase: only {vram_increase:.1f} MB")
    
    if gtt_increase > 1000:  # More than 1GB GTT increase
        print(f"✅ GTT allocation detected! GTT increased by {gtt_increase/1024:.1f} GB")
    else:
        print(f"⚠️ Low GTT increase: only {gtt_increase:.1f} MB")
    
    # Test generation
    print("\n🔥 Testing token generation...")
    test_input = [1, 2, 3, 4, 5]
    
    gen_start = time.time()
    output = pipeline.generate_tokens(test_input, max_tokens=10)
    gen_time = time.time() - gen_start
    
    print(f"\n✅ Generation successful!")
    print(f"   Input: {test_input}")
    print(f"   Output: {output}")
    print(f"   Time: {gen_time:.2f}s")
    print(f"   Speed: {10/gen_time:.1f} tokens/second")
    
    # Final memory check
    final = monitor_memory()
    print(f"\n📊 Final Memory:")
    print(f"   VRAM: {final['vram_mb']:.1f} MB")
    print(f"   GTT: {final['gtt_mb']:.1f} MB")
    print(f"   RAM: {final['ram_gb']:.1f} GB")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    signal.alarm(0)  # Disable timeout
    print("\n✅ Test complete")