#!/usr/bin/env python3
"""Test GPU loading with smaller Gemma 2B model"""

import logging
import subprocess
import time
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

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
                import re
                match = re.search(r'vram\s+[\d.]+%\s+([\d.]+)mb', line)
                if match:
                    vram_mb = float(match.group(1))
            if 'gtt' in line.lower():
                match = re.search(r'gtt\s+[\d.]+%\s+([\d.]+)mb', line)
                if match:
                    gtt_mb = float(match.group(1))
        return vram_mb, gtt_mb
    except:
        return 0, 0

def test_small_model():
    """Test with Gemma 2B model"""
    # Get initial memory
    initial_vram, initial_gtt = get_gpu_memory()
    logger.info(f"📊 Initial GPU Memory: VRAM={initial_vram:.1f}MB, GTT={initial_gtt:.1f}MB")
    
    # Test with 2B model
    logger.info("🚀 Testing with Gemma-2-2B (much smaller, easier to debug)...")
    pipeline = PureHardwarePipelineFixed()
    
    start_time = time.time()
    success = pipeline.initialize(
        model_path="/home/ucadmin/models/gemma-2-2b-it"
    )
    elapsed = time.time() - start_time
    
    if success:
        # Get final memory
        final_vram, final_gtt = get_gpu_memory()
        logger.info(f"📊 Final GPU Memory: VRAM={final_vram:.1f}MB, GTT={final_gtt:.1f}MB")
        
        # Calculate changes
        vram_increase = (final_vram - initial_vram) / 1024  # GB
        gtt_increase = (final_gtt - initial_gtt) / 1024
        total_increase = vram_increase + gtt_increase
        
        logger.info(f"📈 Memory Increase: VRAM={vram_increase:.2f}GB, GTT={gtt_increase:.2f}GB, Total={total_increase:.2f}GB")
        logger.info(f"⏱️  Loading took {elapsed:.1f} seconds")
        
        if total_increase > 1.5:  # 2B model should be ~2GB
            logger.info("✅ SUCCESS! GPU loading works with small model!")
            logger.info("   Now we know the pipeline works - just need to scale to 27B")
            
            # Test a simple inference
            logger.info("\n🧪 Testing inference...")
            start = time.time()
            output = pipeline.generate_tokens([1, 2, 3], max_tokens=10)
            inference_time = time.time() - start
            logger.info(f"   Generated {len(output)} tokens in {inference_time:.2f}s")
            logger.info(f"   TPS: {len(output)/inference_time:.1f}")
        else:
            logger.error("❌ GPU loading still not working properly")
    else:
        logger.error("❌ Pipeline initialization failed")

if __name__ == "__main__":
    test_small_model()