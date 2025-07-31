#!/usr/bin/env python3
"""
Benchmark REAL Gemma3n E4B with NPU+iGPU
Testing actual hardware performance
"""

import time
import logging
from gemma3n_e4b_production_ready import Gemma3nE4BProductionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_real_performance():
    """Test real NPU+iGPU performance with Gemma3n E4B"""
    
    logger.info("🚀 REAL GEMMA3N E4B BENCHMARK - NPU+iGPU HARDWARE")
    logger.info("=" * 60)
    
    # Initialize the production model
    logger.info("⚡ Initializing Gemma3n E4B with real hardware acceleration...")
    
    model = Gemma3nE4BProductionModel(
        model_path="./models/gemma-3n-e4b-it",
        enable_npu=True,
        enable_vulkan=True,
        enable_hma=True,
        turbo_mode=True
    )
    
    # Check if model loaded successfully
    if not hasattr(model, 'model_loaded') or not model.model_loaded:
        logger.warning("Model may not be fully loaded, continuing anyway...")
    
    logger.info("✅ Model initialized")
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "Write a haiku about AI",
        "Explain quantum computing in one sentence",
        "List three colors of the rainbow",
        "What is the capital of France?"
    ]
    
    # Warm-up
    logger.info("\n🔥 Warming up hardware...")
    try:
        _ = model.generate("Hello", max_tokens=5)
        logger.info("✅ Warm-up complete")
    except Exception as e:
        logger.warning(f"⚠️ Warm-up failed: {e}")
    
    # Benchmark
    logger.info("\n📊 Running benchmark...")
    results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nTest {i+1}/{len(test_prompts)}: '{prompt}'")
        
        try:
            start_time = time.time()
            result = model.generate(prompt, max_tokens=50, temperature=0.7)
            elapsed = time.time() - start_time
            
            if isinstance(result, dict):
                tokens_generated = result.get('tokens_generated', 0)
                tps = tokens_generated / elapsed if elapsed > 0 else 0
                
                logger.info(f"  Response: {result.get('generated_text', 'N/A')[:80]}...")
                logger.info(f"  Tokens: {tokens_generated}")
                logger.info(f"  Time: {elapsed:.2f}s")
                logger.info(f"  TPS: {tps:.2f}")
                
                results.append({
                    'prompt': prompt,
                    'tokens': tokens_generated,
                    'time': elapsed,
                    'tps': tps
                })
            else:
                logger.info(f"  Response: {str(result)[:80]}...")
                
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    # Summary
    if results:
        avg_tps = sum(r['tps'] for r in results) / len(results)
        total_tokens = sum(r['tokens'] for r in results)
        total_time = sum(r['time'] for r in results)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Tests completed: {len(results)}/{len(test_prompts)}")
        logger.info(f"✅ Total tokens generated: {total_tokens}")
        logger.info(f"✅ Total time: {total_time:.2f}s")
        logger.info(f"🚀 Average TPS: {avg_tps:.2f} tokens/second")
        logger.info("=" * 60)
        
        # Performance analysis
        if avg_tps >= 150:
            logger.info("🎉 EXCELLENT! Achieved target of 150+ TPS!")
        elif avg_tps >= 100:
            logger.info("✅ Good performance! Over 100 TPS")
        elif avg_tps >= 50:
            logger.info("📈 Decent performance. Room for optimization")
        else:
            logger.info("⚠️ Performance below expectations")
    
    # Show hardware status
    try:
        status = model.get_status()
        logger.info(f"\n📊 Hardware Status: {status}")
    except:
        pass
    
    return avg_tps if results else 0

if __name__ == "__main__":
    try:
        tps = benchmark_real_performance()
        logger.info(f"\n✅ Benchmark completed! Real TPS: {tps:.2f}")
    except Exception as e:
        logger.error(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()