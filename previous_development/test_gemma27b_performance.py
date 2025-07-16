#!/usr/bin/env python3
"""
Test Gemma 3 27B Performance with Real Hardware
Focus on the 27B model as intended and measure actual TPS
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma27BPerformanceTest:
    """Performance testing for Gemma 3 27B with real hardware"""
    
    def __init__(self):
        self.model_paths = {
            "original": "./models/gemma-3-27b-it",
            "ultra_quantized": "./quantized_models/gemma-3-27b-it-ultra-16gb",
            "memory_efficient": "./quantized_models/gemma-3-27b-it-memory-efficient",
            "real_optimized": "./quantized_models/gemma-3-27b-it-real-optimized",
            "vulkan_accelerated": "./quantized_models/gemma-3-27b-it-vulkan-accelerated"
        }
        
        self.test_prompts = [
            "Explain artificial intelligence",
            "What is quantum computing?",
            "How does machine learning work?",
            "Describe renewable energy benefits",
            "What is the future of technology?"
        ]
        
        self.results = {}
        
    def test_model_variant(self, variant_name: str, model_path: str, max_tokens: int = 50) -> Dict:
        """Test a specific model variant"""
        logger.info(f"🧪 Testing {variant_name}: {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"❌ Model not found: {model_path}")
            return {"error": "Model not found"}
        
        try:
            # Load model with optimizations
            logger.info("📥 Loading model...")
            start_load = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            load_time = time.time() - start_load
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            model_size_gb = param_count * 2 / (1024**3)  # Assuming fp16
            
            logger.info(f"   ✅ Model loaded: {param_count:,} parameters ({model_size_gb:.1f}GB)")
            logger.info(f"   ⏱️ Load time: {load_time:.1f}s")
            
            # Performance test
            inference_results = []
            
            for i, prompt in enumerate(self.test_prompts):
                logger.info(f"   🔄 Test {i+1}/5: {prompt[:30]}...")
                
                start_time = time.time()
                
                try:
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_length = inputs.input_ids.shape[1]
                    
                    # Generate with proper settings
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                    
                    # Decode
                    generated_text = tokenizer.decode(
                        outputs[0][input_length:], 
                        skip_special_tokens=True
                    )
                    
                    total_time = time.time() - start_time
                    actual_tokens = len(tokenizer.encode(generated_text))
                    tps = actual_tokens / total_time if total_time > 0 else 0
                    
                    inference_results.append({
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "input_tokens": input_length,
                        "output_tokens": actual_tokens,
                        "total_time": total_time,
                        "tokens_per_second": tps,
                        "success": True
                    })
                    
                    logger.info(f"      ✅ {actual_tokens} tokens in {total_time:.2f}s = {tps:.1f} TPS")
                    
                except Exception as e:
                    logger.error(f"      ❌ Inference failed: {e}")
                    inference_results.append({
                        "prompt": prompt,
                        "error": str(e),
                        "success": False
                    })
                
                # Brief pause between tests
                time.sleep(1)
            
            # Calculate averages
            successful_tests = [r for r in inference_results if r.get("success", False)]
            
            if successful_tests:
                avg_tps = sum(r["tokens_per_second"] for r in successful_tests) / len(successful_tests)
                avg_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
                total_tokens = sum(r["output_tokens"] for r in successful_tests)
                
                logger.info(f"   📊 Average TPS: {avg_tps:.1f}")
                logger.info(f"   📊 Average time: {avg_time:.2f}s")
                logger.info(f"   📊 Success rate: {len(successful_tests)}/{len(inference_results)}")
            else:
                avg_tps = 0
                avg_time = 0
                total_tokens = 0
                logger.error(f"   ❌ No successful inference tests")
            
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "variant_name": variant_name,
                "model_path": model_path,
                "parameter_count": param_count,
                "model_size_gb": model_size_gb,
                "load_time": load_time,
                "inference_results": inference_results,
                "avg_tps": avg_tps,
                "avg_time": avg_time,
                "total_tokens": total_tokens,
                "success_rate": len(successful_tests) / len(inference_results) if inference_results else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return {"error": str(e)}
    
    def run_complete_benchmark(self) -> Dict:
        """Run complete benchmark across all model variants"""
        logger.info("🚀 Starting Gemma 3 27B Performance Benchmark")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Test each available model variant
        for variant_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                result = self.test_model_variant(variant_name, model_path)
                self.results[variant_name] = result
                logger.info("-" * 40)
            else:
                logger.info(f"⚠️ Skipping {variant_name}: not found")
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎯 GEMMA 3 27B PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        successful_variants = {k: v for k, v in self.results.items() if "avg_tps" in v}
        
        if successful_variants:
            # Sort by performance
            sorted_variants = sorted(
                successful_variants.items(), 
                key=lambda x: x[1]["avg_tps"], 
                reverse=True
            )
            
            for variant_name, result in sorted_variants:
                logger.info(f"   🏆 {variant_name}:")
                logger.info(f"      📊 Average TPS: {result['avg_tps']:.1f}")
                logger.info(f"      💾 Model size: {result['model_size_gb']:.1f}GB")
                logger.info(f"      ⏱️ Load time: {result['load_time']:.1f}s")
                logger.info(f"      ✅ Success rate: {result['success_rate']*100:.1f}%")
                
            # Best performer
            best_variant, best_result = sorted_variants[0]
            logger.info(f"")
            logger.info(f"🥇 BEST PERFORMER: {best_variant}")
            logger.info(f"   🚀 {best_result['avg_tps']:.1f} TPS")
            logger.info(f"   📊 {best_result['parameter_count']:,} parameters")
            
        else:
            logger.error("❌ No successful model variants tested")
        
        logger.info(f"")
        logger.info(f"⏱️ Total benchmark time: {total_time:.1f}s")
        logger.info("=" * 60)
        
        return {
            "results": self.results,
            "total_time": total_time,
            "best_variant": best_variant if successful_variants else None,
            "best_tps": best_result["avg_tps"] if successful_variants else 0
        }

def main():
    """Main function"""
    logger.info("🦄 Gemma 3 27B Performance Testing")
    
    # Create and run benchmark
    benchmark = Gemma27BPerformanceTest()
    results = benchmark.run_complete_benchmark()
    
    # Return appropriate exit code
    if results["best_tps"] > 0:
        logger.info("✅ Benchmark completed successfully!")
        return 0
    else:
        logger.error("❌ Benchmark failed - no successful tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())