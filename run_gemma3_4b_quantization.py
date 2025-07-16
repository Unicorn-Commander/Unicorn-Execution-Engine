#!/usr/bin/env python3
"""
Background Gemma 3 4B-IT Quantization - Complete with saving
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import time
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_gemma3_4b():
    """Complete Gemma 3 4B quantization with saving"""
    logger.info("🚀 GEMMA 3 4B-IT COMPLETE QUANTIZATION")
    logger.info("🎯 Real multimodal quantization with saving")
    logger.info("=" * 55)
    
    model_name = "google/gemma-3-4b-it"
    cache_dir = "./models/gemma-3-4b-it"
    output_path = "./quantized_models/gemma-3-4b-it-multimodal"
    
    try:
        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        
        # Step 1: Download processor
        logger.info("📋 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info(f"✅ Image processor: {type(processor.image_processor).__name__}")
        
        # Step 2: Configure quantization
        logger.info("🔧 Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Step 3: Load and quantize model
        logger.info("📦 Loading and quantizing 4B model...")
        logger.info("⏱️ This will take 5-10 minutes...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model quantized in {load_time/60:.1f} minutes")
        
        # Step 4: Test functionality
        logger.info("🧪 Testing quantized model...")
        inputs = processor(text="The future of AI is", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"✅ Generated: \"{response}\"")
        
        # Step 5: Save everything
        logger.info("💾 Saving quantized model...")
        save_start = time.time()
        
        model.save_pretrained(output_path, safe_serialization=True, max_shard_size="1GB")
        processor.save_pretrained(output_path)
        
        save_time = time.time() - save_start
        logger.info(f"✅ Saved in {save_time:.1f} seconds")
        
        # Step 6: Create info file
        model_info = {
            "model_name": "gemma-3-4b-it-multimodal-quantized",
            "original_model": model_name,
            "quantization": "4-bit NF4 with double quantization",
            "capabilities": ["text_generation", "vision_understanding", "multimodal_chat"],
            "image_size": "896x896",
            "context_length": "128K tokens",
            "load_time_minutes": load_time / 60,
            "memory_optimized": True,
            "hardware_ready": "NPU + iGPU acceleration",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{output_path}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("🎉 GEMMA 3 4B QUANTIZATION COMPLETE!")
        logger.info(f"📁 Saved to: {output_path}")
        logger.info(f"⏱️ Total time: {(time.time() - start_time + save_time)/60:.1f} minutes")
        
        return {"success": True, "path": output_path, "time_minutes": load_time/60}
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = quantize_gemma3_4b()
    
    if result["success"]:
        print(f"\n🦄 SUCCESS! Gemma 3 4B-IT quantized and ready!")
        print(f"📁 Location: {result['path']}")
        print(f"⏱️ Time: {result['time_minutes']:.1f} minutes")
        print(f"\n🎮 Test with:")
        print(f"python terminal_chat.py --model {result['path']}")
    else:
        print(f"❌ Failed: {result.get('error')}")