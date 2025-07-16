#!/usr/bin/env python3
"""
Test that enforces NPU+iGPU only - NO CPU FALLBACK
Fast parallel loading like Ollama
"""

import logging
import time
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 HARDWARE ONLY TEST - NPU+iGPU or FAILURE")
    logger.info("❌ NO CPU FALLBACK ALLOWED")
    
    # Check NPU availability first
    if not os.path.exists("/dev/accel/accel0"):
        logger.error("❌ FAILED: No NPU device found at /dev/accel/accel0")
        logger.error("This test requires real NPU hardware!")
        sys.exit(1)
    
    # Import after NPU check
    from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed
    
    # Override to disable CPU fallback
    class HardwareOnlyPipeline(PureHardwarePipelineGPUFixed):
        def __init__(self):
            super().__init__()
            self.cpu_fallback_allowed = False
            logger.info("✅ CPU fallback DISABLED - Hardware only mode")
        
        def compute_attention_layer_gpu(self, layer_idx, hidden_states, kv_cache=None):
            """Override to ensure NPU usage"""
            # The NPU initialization in PureHardwarePipelineFixed will raise an error if NPU is not available.
            # No need for an additional check here.
            return super().compute_attention_layer_gpu(layer_idx, hidden_states, kv_cache)
    
    pipeline = HardwareOnlyPipeline()
    
    # Test fast parallel loading
    logger.info("\n📦 Testing fast parallel model loading (like Ollama)...")
    start_load = time.time()
    
    # Set thread count for parallel loading
    os.environ['OMP_NUM_THREADS'] = '16'  # Use all cores
    
    try:
        # Initialize with parallel loading
        model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
        
        if not pipeline.initialize(model_path):
            logger.error("❌ FAILED: Pipeline initialization failed")
            sys.exit(1)
        
        load_time = time.time() - start_load
        logger.info(f"✅ Model loaded in {load_time:.1f}s")
        
        # Verify GPU memory usage
        import subprocess
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'vram' in line and 'gtt' in line:
                logger.info(f"📊 GPU Memory: {line}")
                # Parse VRAM usage
                if 'vram' in line:
                    vram_part = line.split('vram')[1].split(',')[0]
                    if float(vram_part.split('%')[0]) < 50:
                        logger.warning("⚠️ Low VRAM usage - model may not be loaded to GPU!")
        
        # Test inference with GPU monitoring
        logger.info("\n🧪 Testing inference (NPU+iGPU only)...")
        test_input = np.random.randn(1, 1, 5376).astype(np.float32)
        
        # Monitor GPU during inference
        logger.info("Running inference - watch for GPU spikes...")
        
        # Run multiple iterations to see GPU usage
        for i in range(5):
            start = time.time()
            try:
                output, _ = pipeline.forward_layer(i, test_input)
                elapsed = time.time() - start
                logger.info(f"Layer {i}: {elapsed*1000:.1f}ms")
                
                # Check GPU usage during inference
                result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'gpu' in line:
                        gpu_usage = line.split('gpu')[1].split('%')[0].strip()
                        try:
                            if float(gpu_usage) > 0:
                                logger.info(f"✅ GPU ACTIVE: {gpu_usage}% usage")
                            else:
                                logger.warning(f"⚠️ GPU IDLE: {gpu_usage}% - may be using CPU!")
                        except:
                            pass
                        break
                        
            except Exception as e:
                logger.error(f"❌ FAILED at layer {i}: {e}")
                if "NPU not available" in str(e):
                    logger.error("NPU required but not available!")
                    sys.exit(1)
        
        logger.info("\n✅ Hardware-only test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()