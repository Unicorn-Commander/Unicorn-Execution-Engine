#!/usr/bin/env python3
"""
Test Lightning Fast Loading - Ollama-style
"""
import time
import logging
from lightning_fast_loader import LightningFastLoader
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄 Testing Lightning Fast Loader (Ollama-style)")
    logger.info("=" * 60)
    
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # Test lightning fast loading
    logger.info(f"⚡ Loading model with ALL CPU cores...")
    start_time = time.time()
    
    try:
        loader = LightningFastLoader(model_path)
        
        # Load model weights
        logger.info("📦 Loading quantized weights...")
        weights = loader.load_model_lightning()
        
        load_time = time.time() - start_time
        logger.info(f"\n✅ Model loaded in {load_time:.1f} seconds!")
        
        # Check what was loaded
        if weights:
            logger.info(f"📊 Loaded {len(weights)} weight tensors")
            
            # Sample some weights
            sample_keys = list(weights.keys())[:5]
            for key in sample_keys:
                if hasattr(weights[key], 'shape'):
                    logger.info(f"  - {key}: {weights[key].shape}")
        
        logger.info("\n🎯 Loading Performance:")
        logger.info(f"  - Time: {load_time:.1f}s (vs 2+ minutes for single-threaded)")
        logger.info(f"  - Speed: {25.9 / load_time:.1f} GB/s")
        logger.info(f"  - Method: Memory-mapped, zero-copy, all CPU cores")
        
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n💭 This is how 'Magic Unicorn Unconventional Technology'")
    logger.info("   delivers Ollama-level performance - unconventional = faster!")

if __name__ == "__main__":
    main()