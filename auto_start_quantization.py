#!/usr/bin/env python3
"""
Auto-Start OPTIMAL Quantization
Monitors download progress and automatically starts quantization when ready
"""
import time
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_ready():
    """Check if Gemma 3 27B-IT download is complete"""
    try:
        from transformers import AutoTokenizer
        
        # Quick tokenizer test (always works if model exists)
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')
        
        # Try to load model config (works when files are downloaded)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('google/gemma-3-27b-it')
        
        return True
    except Exception:
        return False

def check_model_files_size():
    """Check actual model file sizes"""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    gemma_dirs = list(cache_dir.glob('*gemma-3-27b-it*')) if cache_dir.exists() else []
    
    total_size_gb = 0
    for gdir in gemma_dirs:
        if gdir.is_dir():
            files = list(gdir.rglob('*.safetensors'))
            total_size_gb += sum(f.stat().st_size for f in files) / (1024**3)
    
    return total_size_gb

def monitor_and_start():
    """Monitor download and auto-start quantization"""
    logger.info("🔍 Monitoring Gemma 3 27B-IT download progress...")
    logger.info("Will auto-start OPTIMAL quantization when ready!")
    logger.info("=" * 60)
    
    check_interval = 30  # Check every 30 seconds
    max_wait_time = 3600  # Max 1 hour wait
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        # Check model readiness
        model_ready = check_model_ready()
        file_size_gb = check_model_files_size()
        
        logger.info(f"⏱️ Elapsed: {elapsed_time//60}m {elapsed_time%60}s | Files: {file_size_gb:.1f}GB | Ready: {model_ready}")
        
        if model_ready and file_size_gb > 40:  # Should be ~50GB when complete
            logger.info("🎉 MODEL DOWNLOAD COMPLETE!")
            logger.info("🚀 Starting OPTIMAL quantization automatically...")
            
            # Start quantization process
            try:
                result = subprocess.run([
                    sys.executable, 'optimal_quantizer.py'
                ], capture_output=False, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ OPTIMAL quantization completed successfully!")
                    
                    # Start quality validation
                    logger.info("🧪 Starting quality validation...")
                    subprocess.run([
                        sys.executable, 'create_quality_validator.py'
                    ], capture_output=False, text=True)
                    
                else:
                    logger.error("❌ Quantization failed")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to start quantization: {e}")
                return False
        
        # Wait and check again
        time.sleep(check_interval)
        elapsed_time += check_interval
        
        # Progress indicators
        if elapsed_time % 120 == 0:  # Every 2 minutes
            logger.info("💡 While waiting, you can review:")
            logger.info("   📋 IMPLEMENTATION_PLAN.md - Complete roadmap")
            logger.info("   🎯 OPTIMAL_STRATEGY.md - 150+ TPS strategy")
            logger.info("   ⚡ optimal_quantizer.py - Ready to execute")
    
    logger.warning("⏰ Maximum wait time reached")
    logger.info("You can manually start quantization with: python optimal_quantizer.py")
    return False

def main():
    """Main monitoring function"""
    logger.info("🦄 Gemma 3 27B OPTIMAL Quantization Auto-Starter")
    logger.info("🎯 Target: 80%+ compression for 150+ TPS performance")
    logger.info("=" * 60)
    
    # Quick system check
    logger.info("📋 Pre-flight check:")
    
    # Check if scripts exist
    required_files = ['optimal_quantizer.py', 'create_quality_validator.py']
    for file in required_files:
        if Path(file).exists():
            logger.info(f"   ✅ {file}")
        else:
            logger.error(f"   ❌ {file} not found")
            return False
    
    # Check hardware
    try:
        result = subprocess.run(['python', 'hardware_checker.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("   ✅ Hardware compatibility verified")
        else:
            logger.warning("   ⚠️ Hardware check had warnings")
    except:
        logger.warning("   ⚠️ Could not verify hardware")
    
    logger.info("\n🚀 Starting automated monitoring...")
    
    # Start monitoring
    success = monitor_and_start()
    
    if success:
        logger.info("\n🎉 PHASE 1 COMPLETE!")
        logger.info("📋 Next steps:")
        logger.info("1. Review quantization results")
        logger.info("2. Validate quality metrics >93%")
        logger.info("3. Begin Phase 2: NPU custom kernels")
    else:
        logger.info("\n⏳ Continue monitoring manually")
        logger.info("Or run: python optimal_quantizer.py when ready")

if __name__ == "__main__":
    main()