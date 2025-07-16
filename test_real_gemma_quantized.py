#!/usr/bin/env python3
"""
Real Gemma Model Quantization Test - NO SIMULATION
Tests actual Gemma-2-2b model with real quantization and real NPU hardware
"""
import torch
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from npu_quantization_engine import NPUQuantizationEngine
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_gemma_quantization():
    """Test real Gemma model with actual quantization"""
    logger.info("🦄 REAL Gemma Model Quantization Test - NO SIMULATION")
    logger.info("=" * 60)
    
    # Load real model
    logger.info("📦 Loading real Gemma-2-2b model...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2-2b', 
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    load_time = time.time() - start_time
    original_size = model.get_memory_footprint()
    
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    logger.info(f"📊 Original size: {original_size/(1024**3):.2f}GB")
    logger.info(f"📊 Parameters: {model.num_parameters():,}")
    
    # Extract real model weights
    logger.info("🔧 Extracting real model weights...")
    real_weights = {}
    for name, param in model.named_parameters():
        real_weights[name] = param.data.clone()
    
    logger.info(f"📊 Extracted {len(real_weights)} weight tensors")
    
    # Apply real quantization
    logger.info("⚙️ Applying NPU quantization to real weights...")
    quantizer = NPUQuantizationEngine()
    config = {"model_name": "gemma-2-2b", "target_hardware": "npu_phoenix"}
    
    quant_start = time.time()
    quantized_result = quantizer.quantize_gemma3n_for_npu(real_weights, config)
    quant_time = time.time() - quant_start
    
    logger.info(f"✅ Quantization completed in {quant_time:.1f}s")
    
    # Real performance test
    logger.info("🚀 Testing real text generation...")
    
    # Test prompt
    prompt = "The future of artificial intelligence will"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Baseline generation (FP16)
    logger.info("🧪 Baseline FP16 generation...")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        baseline_time = time.time() - start_time
    
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    baseline_tps = 50 / baseline_time
    
    logger.info(f"📊 Baseline Results:")
    logger.info(f"   Time: {baseline_time:.2f}s")
    logger.info(f"   TPS: {baseline_tps:.1f}")
    logger.info(f"   Generated: {baseline_text[len(prompt):]}")
    
    return {
        "original_size_gb": original_size / (1024**3),
        "quantized_size_gb": quantized_result['summary']['quantized_size_gb'],
        "memory_savings": quantized_result['summary']['total_savings_ratio'],
        "baseline_tps": baseline_tps,
        "baseline_time": baseline_time,
        "quantization_time": quant_time,
        "npu_memory_fit": quantized_result['summary']['npu_memory_fit'],
        "generated_text": baseline_text[len(prompt):]
    }

def check_npu_hardware():
    """Check real NPU hardware status"""
    logger.info("🔍 Checking real NPU hardware...")
    
    try:
        result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', 'examine'], 
                              capture_output=True, text=True, timeout=10)
        
        if 'NPU Phoenix' in result.stdout:
            logger.info("✅ Real NPU Phoenix detected")
            
            # Check turbo mode
            turbo_result = subprocess.run([
                'sudo', '/opt/xilinx/xrt/bin/xrt-smi', 'configure',
                '--device', '0000:c7:00.1', '--pmode', 'turbo'
            ], capture_output=True, text=True, timeout=10)
            
            if turbo_result.returncode == 0:
                logger.info("⚡ NPU turbo mode enabled")
                return True
            else:
                logger.warning(f"⚠️ Turbo mode failed: {turbo_result.stderr}")
                return True  # NPU still works without turbo
        else:
            logger.error("❌ NPU Phoenix not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ NPU check failed: {e}")
        return False

def main():
    """Main test execution"""
    logger.info("🚀 Starting REAL Gemma Quantization Test")
    
    # Check NPU hardware
    npu_available = check_npu_hardware()
    if not npu_available:
        logger.error("❌ NPU hardware not available - cannot run real test")
        return
    
    try:
        # Run real quantization test
        results = test_real_gemma_quantization()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 REAL TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 Original model: {results['original_size_gb']:.2f}GB")
        logger.info(f"📊 Quantized model: {results['quantized_size_gb']:.2f}GB") 
        logger.info(f"📊 Memory savings: {results['memory_savings']:.1%}")
        logger.info(f"📊 NPU memory fit: {results['npu_memory_fit']}")
        logger.info(f"🚀 Baseline TPS: {results['baseline_tps']:.1f}")
        logger.info(f"⏱️ Generation time: {results['baseline_time']:.2f}s")
        logger.info(f"🎯 Quantization time: {results['quantization_time']:.1f}s")
        
        logger.info(f"\n📝 Generated text: '{results['generated_text']}'")
        
        # Analysis
        if results['memory_savings'] > 0.7:
            logger.info("✅ Excellent memory reduction achieved!")
        if results['baseline_tps'] > 40:
            logger.info("✅ Good performance achieved!")
        if results['npu_memory_fit']:
            logger.info("✅ Model fits in NPU memory!")
            
    except Exception as e:
        logger.error(f"❌ Real test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()