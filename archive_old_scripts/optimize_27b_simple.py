#!/usr/bin/env python3
"""
Simple but effective 27B optimization
Focus on moving computation off CPU using automatic device mapping
"""
import torch
import time
import logging
import os
from transformers import AutoProcessor, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_27b_simple():
    """Simple optimization to move 27B computation off CPU"""
    logger.info("🦄 SIMPLE 27B OPTIMIZATION")
    logger.info("🎯 Goal: Move computation OFF CPU onto NPU+iGPU")
    logger.info("=" * 50)
    
    model_path = "./models/gemma-3-27b-it"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 27B model not found: {model_path}")
        return None
    
    try:
        # Load model with automatic device mapping (no quantization conflicts)
        logger.info("📦 Loading 27B with automatic optimal device mapping...")
        load_start = time.time()
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Use auto device mapping for optimal placement
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # Automatic optimal placement
            torch_dtype=torch.float16,  # FP16 for speed
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        load_time = time.time() - load_start
        logger.info(f"✅ Model loaded in {load_time/60:.1f} minutes")
        
        # Check device placement
        logger.info("🔍 Checking device placement...")
        device_usage = {"cpu": 0, "cuda": 0}
        total_params = 0
        
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                device_str = str(param.device).lower()
                if 'cuda' in device_str:
                    device_usage["cuda"] += 1
                else:
                    device_usage["cpu"] += 1
                total_params += 1
        
        logger.info(f"   📊 Total parameters: {total_params}")
        logger.info(f"   🎮 iGPU (CUDA): {device_usage['cuda']} ({device_usage['cuda']/total_params*100:.1f}%)")
        logger.info(f"   💻 CPU: {device_usage['cpu']} ({device_usage['cpu']/total_params*100:.1f}%)")
        
        # Performance test
        logger.info("🚀 Testing optimized performance...")
        
        test_prompts = [
            "The future of AI is",
            "Quantum computing will revolutionize",
            "Renewable energy technologies are advancing"
        ]
        
        results = []
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"   Test {i+1}/3: '{prompt}'")
            
            # Process input
            inputs = processor(text=prompt, return_tensors="pt")
            
            # Move inputs to model device if needed
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate
            gen_start = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # Greedy for speed
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            gen_time = time.time() - gen_start
            
            # Calculate metrics
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            tps = output_tokens / gen_time if gen_time > 0 else 0
            
            total_tokens += output_tokens
            total_time += gen_time
            
            # Decode response
            response = processor.decode(outputs[0][input_tokens:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "tokens": output_tokens,
                "time": gen_time,
                "tps": tps,
                "response": response[:50] + "..." if len(response) > 50 else response
            })
            
            logger.info(f"     ✅ {output_tokens} tokens in {gen_time:.1f}s → {tps:.1f} TPS")
        
        # Overall performance
        optimized_tps = total_tokens / total_time if total_time > 0 else 0
        baseline_tps = 0.9
        improvement = optimized_tps / baseline_tps if baseline_tps > 0 else 1
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 SIMPLE OPTIMIZATION RESULTS")
        logger.info(f"📊 Baseline TPS: {baseline_tps:.1f}")
        logger.info(f"🚀 Optimized TPS: {optimized_tps:.1f}")
        logger.info(f"📈 Improvement: {improvement:.1f}x faster")
        logger.info(f"📊 Total tokens: {total_tokens}")
        logger.info(f"⏱️ Total time: {total_time:.1f}s")
        
        # Device utilization summary
        logger.info("\n🔧 DEVICE UTILIZATION:")
        if device_usage["cuda"] > device_usage["cpu"]:
            logger.info("   ✅ MOSTLY iGPU: Computation moved to GPU successfully")
        elif device_usage["cuda"] > 0:
            logger.info("   📊 HYBRID: Both CPU and iGPU utilized")
        else:
            logger.info("   ⚠️ CPU ONLY: Need to enable GPU acceleration")
        
        # Performance assessment
        if optimized_tps >= 5.0:
            logger.info("✅ GOOD: 5+ TPS achieved, ready for Vulkan acceleration")
        elif optimized_tps >= 2.0:
            logger.info("📈 PROGRESS: 2+ TPS, on track for optimization")
        else:
            logger.info("⚠️ SLOW: Need more aggressive optimization")
        
        return {
            "success": True,
            "optimized_tps": optimized_tps,
            "improvement": improvement,
            "device_usage": device_usage,
            "gpu_percentage": device_usage["cuda"] / total_params * 100,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = optimize_27b_simple()
    
    if result and result["success"]:
        print(f"\n🦄 SIMPLE 27B OPTIMIZATION SUCCESS!")
        print(f"🚀 Optimized performance: {result['optimized_tps']:.1f} TPS")
        print(f"📈 Improvement: {result['improvement']:.1f}x faster")
        print(f"🎮 GPU utilization: {result['gpu_percentage']:.1f}%")
        
        if result['gpu_percentage'] > 50:
            print("✅ SUCCESSFUL: Computation moved to GPU!")
        else:
            print("📊 PARTIAL: Some computation on GPU")
    else:
        print(f"❌ Optimization failed: {result.get('error') if result else 'Unknown error'}")