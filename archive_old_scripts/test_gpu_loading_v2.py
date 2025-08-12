#!/usr/bin/env python3
"""Test the fixed GPU loading"""

import logging
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current VRAM and GTT usage"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=2)
        vram_mb = 0
        gtt_mb = 0
        for line in result.stdout.split('\n'):
            if 'vram' in line.lower():
                # Extract VRAM in MB
                import re
                match = re.search(r'vram\s+[\d.]+%\s+([\d.]+)mb', line)
                if match:
                    vram_mb = float(match.group(1))
            if 'gtt' in line.lower():
                # Extract GTT in MB
                match = re.search(r'gtt\s+[\d.]+%\s+([\d.]+)mb', line)
                if match:
                    gtt_mb = float(match.group(1))
        return vram_mb, gtt_mb
    except:
        return 0, 0

def test_loading():
    """Test GPU loading with the fixed pipeline"""
    # Get initial memory
    initial_vram, initial_gtt = get_gpu_memory()
    logger.info(f"📊 Initial GPU Memory: VRAM={initial_vram:.1f}MB, GTT={initial_gtt:.1f}MB")
    
    # Import and test the fixed pipeline
    from pure_hardware_pipeline_fixed_v2 import PureHardwarePipelineFixed
    
    logger.info("🚀 Initializing fixed pipeline...")
    pipeline = PureHardwarePipelineFixed()
    
    # Initialize with timeout
    start_time = time.time()
    success = pipeline.initialize(
        model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    )
    
    elapsed = time.time() - start_time
    logger.info(f"⏱️  Initialization took {elapsed:.1f} seconds")
    
    if success:
        # Get final memory
        final_vram, final_gtt = get_gpu_memory()
        logger.info(f"📊 Final GPU Memory: VRAM={final_vram:.1f}MB, GTT={final_gtt:.1f}MB")
        
        # Calculate changes
        vram_increase = (final_vram - initial_vram) / 1024  # Convert to GB
        gtt_increase = (final_gtt - initial_gtt) / 1024
        
        logger.info(f"📈 Memory Increase: VRAM={vram_increase:.2f}GB, GTT={gtt_increase:.2f}GB")
        
        if vram_increase > 10:  # Should be ~16GB
            logger.info("✅ SUCCESS! GPU loading is working!")
            logger.info(f"   Model loaded to GPU: {vram_increase + gtt_increase:.1f}GB total")
        else:
            logger.error("❌ FAIL: GPU memory did not increase as expected")
            logger.info("   The model is NOT loading to GPU properly")
    else:
        logger.error("❌ Pipeline initialization failed")

if __name__ == "__main__":
    test_loading()