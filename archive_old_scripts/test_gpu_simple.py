#!/usr/bin/env python3
"""
Simple GPU test to verify GPU compute is actually working
"""

import logging
import subprocess
import time
import threading
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to stop monitoring
stop_monitoring = False

def monitor_gpu():
    """Monitor GPU usage in background"""
    global stop_monitoring
    while not stop_monitoring:
        try:
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'gpu' in line:
                    # Extract GPU usage percentage
                    parts = line.split(',')
                    for part in parts:
                        if 'gpu' in part:
                            logger.info(f"🎮 GPU: {part.strip()}")
                            break
        except:
            pass
        time.sleep(1)

def main():
    logger.info("🎯 Simple GPU compute test")
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineGPUFixed()
    
    # Just test matrix multiply
    logger.info("Testing GPU compute...")
    
    try:
        # Initialize Vulkan
        if pipeline.initialize("/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"):
            logger.info("✅ Model loaded to GPU")
            
            # Simple forward pass test
            import numpy as np
            test_input = np.random.randn(1, 1, 5376).astype(np.float32)
            
            logger.info("Running forward pass...")
            start = time.time()
            
            # Just one layer
            output, _ = pipeline.forward_layer(0, test_input)
            
            elapsed = time.time() - start
            logger.info(f"✅ Forward pass completed in {elapsed*1000:.2f}ms")
            
            # Test a few more layers
            for i in range(5):
                logger.info(f"Testing layer {i}...")
                output, _ = pipeline.forward_layer(i, test_input)
                
        else:
            logger.error("Failed to initialize")
            
    finally:
        # Stop monitoring
        global stop_monitoring
        stop_monitoring = True
        monitor_thread.join(timeout=2)
        
        # Cleanup
        if pipeline:
            pipeline.cleanup()

if __name__ == "__main__":
    main()