#!/usr/bin/env python3
"""
Gemma 3 27B Baseline Performance Benchmark
Establishes performance baseline before optimization
"""
import torch
import time
import psutil
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import json
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma327BBaseline:
    """Baseline performance testing for Gemma 3 27B"""
    
    def __init__(self, model_id: str = "google/gemma-3-27b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.baseline_results = {}
        
    def load_model(self) -> bool:
        """Load Gemma 3 27B model"""
        logger.info(f"📦 Loading {self.model_id}...")
        
        try:
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info(f"✅ Tokenizer loaded: {self.tokenizer.vocab_size:,} tokens")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            model_size_gb = self.model.get_memory_footprint() / (1024**3)
            
            logger.info(f"✅ Model loaded in {load_time/60:.1f} minutes")
            logger.info(f"📊 Model size: {model_size_gb:.1f}GB")
            logger.info(f"📊 Parameters: {self.model.num_parameters():,}")
            
            self.baseline_results.update({
                "load_time_minutes": load_time / 60,
                "model_size_gb": model_size_gb,
                "parameters": self.model.num_parameters(),
                "tokenizer_vocab_size": self.tokenizer.vocab_size
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def run_memory_analysis(self) -> Dict[str, float]:
        """Analyze memory usage patterns"""
        logger.info("💾 Running memory analysis...")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_analysis = {
            "rss_gb": memory_info.rss / (1024**3),
            "vms_gb": memory_info.vms / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "memory_percent": process.memory_percent()
        }
        
        logger.info(f"📊 Memory Analysis:")
        for key, value in memory_analysis.items():
            logger.info(f"   {key}: {value:.2f}{'%' if 'percent' in key else 'GB' if 'gb' in key else ''}")
        
        return memory_analysis
    
    def run_generation_benchmarks(self) -> Dict[str, List[Dict]]:
        """Run comprehensive generation benchmarks"""
        logger.info("🚀 Running generation benchmarks...")
        
        test_scenarios = [
            {
                "name": "Quick Response",
                "prompt": "AI will",
                "max_tokens": 10,
                "description": "Minimal generation test"
            },
            {
                "name": "Short Generation", 
                "prompt": "The future of artificial intelligence",
                "max_tokens": 25,
                "description": "Short response test"
            },
            {
                "name": "Medium Generation",
                "prompt": "Explain the benefits of quantum computing in simple terms",
                "max_tokens": 50,
                "description": "Medium response test"
            },
            {
                "name": "Long Generation",
                "prompt": "Write a comprehensive analysis of machine learning trends",
                "max_tokens": 100,
                "description": "Long response test"
            }
        ]
        
        benchmark_results = []
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"🧪 Test {i+1}/4: {scenario['name']}")
            logger.info(f"   Prompt: '{scenario['prompt']}'")
            logger.info(f"   Max tokens: {scenario['max_tokens']}")
            
            try:
                # Prepare inputs
                inputs = self.tokenizer(scenario['prompt'], return_tensors="pt")
                input_length = inputs.input_ids.shape[1]
                
                # Memory before generation
                gc.collect()
                memory_before = psutil.Process().memory_info().rss / (1024**3)
                
                # Generation
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=scenario['max_tokens'],
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                generation_time = time.time() - start_time
                
                # Memory after generation
                memory_after = psutil.Process().memory_info().rss / (1024**3)
                memory_delta = memory_after - memory_before
                
                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_tokens = len(outputs[0]) - input_length
                
                # Calculate metrics
                tps = generated_tokens / generation_time
                ttft = generation_time / generated_tokens if generated_tokens > 0 else 0  # Simplified TTFT
                
                result = {
                    "scenario": scenario['name'],
                    "prompt": scenario['prompt'],
                    "generated_text": generated_text[len(scenario['prompt']):].strip(),
                    "input_tokens": input_length,
                    "generated_tokens": generated_tokens,
                    "total_tokens": len(outputs[0]),
                    "generation_time_seconds": generation_time,
                    "tokens_per_second": tps,
                    "time_per_token_ms": (generation_time / generated_tokens * 1000) if generated_tokens > 0 else 0,
                    "memory_delta_gb": memory_delta,
                    "memory_efficiency": generated_tokens / memory_delta if memory_delta > 0 else 0
                }
                
                benchmark_results.append(result)
                
                logger.info(f"✅ Results:")
                logger.info(f"   Generated: {generated_tokens} tokens")
                logger.info(f"   Time: {generation_time:.2f}s")
                logger.info(f"   TPS: {tps:.2f}")
                logger.info(f"   Memory: +{memory_delta:.2f}GB")
                logger.info(f"   Text: '{generated_text[len(scenario['prompt']):].strip()[:50]}...'")
                
            except Exception as e:
                logger.error(f"❌ Test {scenario['name']} failed: {e}")
                benchmark_results.append({
                    "scenario": scenario['name'],
                    "error": str(e)
                })
            
            logger.info("")  # Blank line between tests
        
        return benchmark_results
    
    def analyze_performance(self, benchmark_results: List[Dict]) -> Dict[str, float]:
        """Analyze overall performance characteristics"""
        logger.info("📊 Analyzing baseline performance...")
        
        valid_results = [r for r in benchmark_results if 'error' not in r]
        
        if not valid_results:
            logger.error("❌ No valid benchmark results to analyze")
            return {}
        
        # Calculate statistics
        tps_values = [r['tokens_per_second'] for r in valid_results]
        memory_deltas = [r['memory_delta_gb'] for r in valid_results]
        generation_times = [r['generation_time_seconds'] for r in valid_results]
        
        performance_analysis = {
            "average_tps": sum(tps_values) / len(tps_values),
            "min_tps": min(tps_values),
            "max_tps": max(tps_values),
            "average_memory_delta_gb": sum(memory_deltas) / len(memory_deltas),
            "total_generation_time": sum(generation_times),
            "successful_tests": len(valid_results),
            "failed_tests": len(benchmark_results) - len(valid_results)
        }
        
        logger.info(f"📈 Performance Summary:")
        logger.info(f"   Average TPS: {performance_analysis['average_tps']:.2f}")
        logger.info(f"   TPS Range: {performance_analysis['min_tps']:.2f} - {performance_analysis['max_tps']:.2f}")
        logger.info(f"   Memory per generation: {performance_analysis['average_memory_delta_gb']:.2f}GB")
        logger.info(f"   Success rate: {performance_analysis['successful_tests']}/{len(benchmark_results)}")
        
        return performance_analysis
    
    def save_baseline_results(self, filename: str = "gemma3_27b_baseline.json"):
        """Save all baseline results to file"""
        logger.info(f"💾 Saving baseline results to {filename}")
        
        with open(filename, 'w') as f:
            json.dump(self.baseline_results, f, indent=2, default=str)
        
        logger.info(f"✅ Baseline results saved")

def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='Gemma 3 27B Baseline Benchmark')
    parser.add_argument('--model', default='google/gemma-3-27b-it', help='Model ID')
    parser.add_argument('--skip-download', action='store_true', help='Skip model download if already cached')
    args = parser.parse_args()
    
    logger.info("🦄 Gemma 3 27B Baseline Performance Benchmark")
    logger.info("=" * 60)
    
    baseline = Gemma327BBaseline(args.model)
    
    try:
        # Load model
        if not baseline.load_model():
            logger.error("❌ Failed to load model")
            return
        
        # Memory analysis
        memory_analysis = baseline.run_memory_analysis()
        baseline.baseline_results['memory_analysis'] = memory_analysis
        
        # Generation benchmarks  
        benchmark_results = baseline.run_generation_benchmarks()
        baseline.baseline_results['benchmark_results'] = benchmark_results
        
        # Performance analysis
        performance_analysis = baseline.analyze_performance(benchmark_results)
        baseline.baseline_results['performance_analysis'] = performance_analysis
        
        # Save results
        baseline.save_baseline_results()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 BASELINE ESTABLISHMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📊 Model: {args.model}")
        logger.info(f"📊 Size: {baseline.baseline_results['model_size_gb']:.1f}GB")
        logger.info(f"📊 Parameters: {baseline.baseline_results['parameters']:,}")
        logger.info(f"🚀 Average TPS: {performance_analysis.get('average_tps', 0):.2f}")
        logger.info(f"💾 Memory usage: {memory_analysis['rss_gb']:.1f}GB")
        
        logger.info("\n🎯 OPTIMIZATION TARGETS:")
        current_tps = performance_analysis.get('average_tps', 0)
        target_tps_min = 113
        target_tps_max = 162
        improvement_needed = target_tps_min / current_tps if current_tps > 0 else 0
        
        logger.info(f"   Current: {current_tps:.1f} TPS")
        logger.info(f"   Target: {target_tps_min}-{target_tps_max} TPS")
        logger.info(f"   Improvement needed: {improvement_needed:.1f}x")
        
        logger.info("\n✅ Ready for Phase 2: Quantization")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()