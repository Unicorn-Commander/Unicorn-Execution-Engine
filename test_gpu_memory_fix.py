#!/usr/bin/env python3
"""Test GPU memory usage with int8 weights"""

import time
import logging
import subprocess
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        # Get VRAM usage
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=2)
        output = result.stdout
        vram_mb = 0
        gtt_mb = 0
        
        for line in output.split('\n'):
            if 'vram' in line.lower():
                # Extract VRAM MB value
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'mb' in part.lower() and i > 0:
                        try:
                            vram_mb = float(parts[i-1])
                        except:
                            pass
            elif 'gtt' in line.lower():
                # Extract GTT MB value
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'mb' in part.lower() and i > 0:
                        try:
                            gtt_mb = float(parts[i-1])
                        except:
                            pass
        
        return vram_mb, gtt_mb
    except:
        return 0, 0

def main():
    """Test GPU memory with fixed loading"""
    logger.info("🧪 Testing GPU Memory Usage Fix")
    
    # Get baseline
    vram_base, gtt_base = get_gpu_memory()
    logger.info(f"📊 Baseline: VRAM {vram_base:.1f}MB, GTT {gtt_base:.1f}MB")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("🔄 Loading model...")
    start_time = time.time()
    
    if pipeline.initialize(model_path):
        elapsed = time.time() - start_time
        
        # Get memory after loading
        vram_after, gtt_after = get_gpu_memory()
        vram_used = vram_after - vram_base
        gtt_used = gtt_after - gtt_base
        
        logger.info(f"✅ Model loaded in {elapsed:.1f}s")
        logger.info(f"📊 GPU Memory Usage:")
        logger.info(f"   VRAM: {vram_used/1024:.1f}GB (was: {vram_base:.1f}MB → {vram_after:.1f}MB)")
        logger.info(f"   GTT:  {gtt_used/1024:.1f}GB (was: {gtt_base:.1f}MB → {gtt_after:.1f}MB)")
        logger.info(f"   Total: {(vram_used + gtt_used)/1024:.1f}GB")
        
        # Expected vs actual
        logger.info(f"\n📋 Analysis:")
        logger.info(f"   Expected for 27B int8: ~26GB")
        logger.info(f"   Actual: {(vram_used + gtt_used)/1024:.1f}GB")
        
        if (vram_used + gtt_used)/1024 > 50:  # More than 50GB
            logger.warning("   ⚠️ Memory usage too high - weights being converted to float32!")
        else:
            logger.info("   ✅ Memory usage looks correct for int8 weights")
        
        # Keep alive to check
        time.sleep(5)
        pipeline.cleanup()
    else:
        logger.error("❌ Pipeline initialization failed")

if __name__ == "__main__":
    main()