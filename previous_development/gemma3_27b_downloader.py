#!/usr/bin/env python3
"""
Gemma 3 27B Model Downloader and Initial Setup
Downloads and prepares Gemma 3 27B for optimization
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import logging
import time
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_gemma3_27b():
    """Download Gemma 3 27B model"""
    logger.info("🚀 Starting Gemma 3 27B Download & Setup")
    logger.info("=" * 60)
    
    model_id = "google/gemma-3-27b-it"  # Gemma 3 instruction tuned variant
    
    # Check available space
    disk_usage = os.statvfs('.')
    free_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
    logger.info(f"💾 Available disk space: {free_space_gb:.1f}GB")
    
    if free_space_gb < 60:
        logger.error("❌ Insufficient disk space. Need 60GB+")
        return False
    
    try:
        # Download config first
        logger.info("📋 Downloading model configuration...")
        config = AutoConfig.from_pretrained(model_id)
        logger.info(f"✅ Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        
        # Download tokenizer
        logger.info("🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info(f"✅ Tokenizer: {tokenizer.vocab_size} tokens")
        
        # Download model (this will take time)
        logger.info("📦 Downloading Gemma 3 27B model... (this may take 30+ minutes)")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",  # Keep on CPU for now
            trust_remote_code=True
        )
        
        download_time = time.time() - start_time
        model_size_gb = model.get_memory_footprint() / (1024**3)
        
        logger.info(f"✅ Model downloaded in {download_time/60:.1f} minutes")
        logger.info(f"📊 Model size: {model_size_gb:.1f}GB")
        logger.info(f"📊 Parameters: {model.num_parameters():,}")
        
        # Quick test
        logger.info("🧪 Running quick generation test...")
        test_prompt = "The future of AI"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            test_start = time.time()
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            test_time = time.time() - test_start
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_tps = 10 / test_time
        
        logger.info(f"📝 Generated: {generated_text}")
        logger.info(f"⏱️ Baseline TPS: {baseline_tps:.1f}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "baseline_tps": baseline_tps,
            "model_size_gb": model_size_gb
        }
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_gemma3_27b()