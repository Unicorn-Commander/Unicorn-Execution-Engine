#!/usr/bin/env python3
"""
Test Magic Unicorn prompt with working GPU pipeline
"""

import logging
import time
from gpu_pipeline_working import GPUPipelineWorking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🦄 Testing Magic Unicorn with WORKING GPU pipeline...")
    
    pipeline = GPUPipelineWorking()
    
    # Initialize
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    logger.info("Loading model...")
    
    start_load = time.time()
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize")
        return
    load_time = time.time() - start_load
    
    logger.info(f"✅ Model loaded in {load_time:.1f}s!")
    
    # Test prompt
    prompt = "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that"
    logger.info(f"\n📝 Prompt: '{prompt}'")
    
    # Simple tokenization
    input_ids = [ord(c) % 1000 for c in prompt]
    logger.info(f"Tokens: {len(input_ids)}")
    
    # Generate
    logger.info("\n🚀 Generating response...")
    start_gen = time.time()
    
    try:
        generated_ids = pipeline.generate_tokens(
            input_ids, 
            max_tokens=20,  # Just 20 tokens for quick test
            temperature=0.7,
            top_p=0.9
        )
        
        gen_time = time.time() - start_gen
        tps = len(generated_ids) / gen_time if gen_time > 0 else 0
        
        logger.info(f"\n✅ Generation complete!")
        logger.info(f"   Generated {len(generated_ids)} tokens in {gen_time:.1f}s")
        logger.info(f"   ⚡ Performance: {tps:.1f} TPS")
        
        # Simple detokenization
        response = ''.join([chr((t % 94) + 33) for t in generated_ids])
        logger.info(f"\n🦄 Generated text: {response}")
        
        # Performance analysis
        logger.info("\n📊 Performance Analysis:")
        if tps >= 81:
            logger.info("   🎉 TARGET ACHIEVED! 81+ TPS!")
        elif tps >= 50:
            logger.info("   ✅ Good performance! On track for optimization")
        elif tps >= 20:
            logger.info("   ⚠️ Moderate performance, needs optimization")
        else:
            logger.info("   ❌ Low performance, investigate bottlenecks")
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Monitor GPU one more time
    import subprocess
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'gpu' in line.lower():
                logger.info(f"\n🎮 Final GPU status: {line}")
                break
    except:
        pass
    
    # Cleanup
    pipeline.cleanup()
    logger.info("\n✅ Test complete!")

if __name__ == "__main__":
    main()