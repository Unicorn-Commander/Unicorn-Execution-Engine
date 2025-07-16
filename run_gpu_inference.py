#!/usr/bin/env python3
"""
Run inference using GPU memory properly
This bypasses the problematic CPU loading
"""

import numpy as np
import time
import logging
import subprocess
from pure_hardware_pipeline import PureHardwarePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_gpu_memory():
    """Get current GPU memory usage"""
    result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                          capture_output=True, text=True)
    output = result.stdout
    if 'vram' in output and 'gtt' in output:
        return output.strip().split('\n')[-1]
    return "GPU monitoring not available"

def main():
    print("🚀 Running GPU Inference Test")
    print("=" * 60)
    
    # Monitor baseline
    print("\n📊 Baseline GPU Memory:")
    print(monitor_gpu_memory())
    
    # Initialize pipeline
    print("\n🔄 Initializing pipeline...")
    pipeline = PureHardwarePipeline()
    
    # Use timeout to prevent hanging
    import signal
    
    def timeout_handler(signum, frame):
        print("\n⏰ Initialization timeout - checking GPU memory...")
        print(monitor_gpu_memory())
        
        # Check if GPU memory increased
        gpu_status = monitor_gpu_memory()
        if 'vram' in gpu_status:
            parts = gpu_status.split(',')
            for part in parts:
                if 'vram' in part and 'mb' in part:
                    vram_str = part.strip().split()[-1]
                    vram_mb = float(vram_str.replace('mb', ''))
                    if vram_mb > 5000:  # More than 5GB VRAM
                        print(f"✅ GPU allocation successful! VRAM: {vram_mb}MB")
                    else:
                        print(f"⚠️ Low VRAM usage: {vram_mb}MB")
        
        raise TimeoutError("Initialization timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # 3 minute timeout
    
    try:
        # Initialize the model
        success = pipeline.initialize('quantized_models/gemma-3-27b-it-layer-by-layer')
        signal.alarm(0)  # Disable timeout
        
        if success:
            print("\n✅ Pipeline initialized!")
            
            # Check GPU memory
            print("\n📊 GPU Memory After Init:")
            print(monitor_gpu_memory())
            
            # Wait a moment to ensure memory is stable
            time.sleep(2)
            
            # Try simple generation
            print("\n🔥 Testing generation...")
            start_time = time.time()
            
            # Test with simple input
            test_input = [1, 2, 3, 4, 5]
            print(f"Input tokens: {test_input}")
            
            # Generate a few tokens
            try:
                output = pipeline.generate_tokens(test_input, max_tokens=10)
                elapsed = time.time() - start_time
                
                print(f"Generated tokens: {output}")
                print(f"Generation time: {elapsed:.2f}s")
                print(f"Tokens per second: {10/elapsed:.1f} TPS")
                
                # Final GPU check
                print("\n📊 Final GPU Memory:")
                print(monitor_gpu_memory())
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("❌ Pipeline initialization failed")
            
    except TimeoutError:
        print("\n⏰ Model loading timed out")
        print("This likely means it's using CPU memory instead of GPU")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        signal.alarm(0)  # Ensure timeout is disabled
        
        # Final memory check
        print("\n📊 Final System Status:")
        
        # GPU memory
        print("GPU: " + monitor_gpu_memory())
        
        # System memory
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.used/1024/1024/1024:.1f}GB / {mem.total/1024/1024/1024:.1f}GB ({mem.percent:.1f}%)")

if __name__ == "__main__":
    # Install psutil if needed
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        subprocess.run(['pip', 'install', 'psutil'])
        import psutil
    
    main()