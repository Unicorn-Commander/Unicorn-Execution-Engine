#!/usr/bin/env python3
"""
Download and convert Gemma models to GGUF format for NPU+iGPU testing
Supports both Gemma 4B and 27B models
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required tools are available"""
    logger.info("🔍 Checking dependencies...")
    
    # Check for llama.cpp convert script
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        logger.error(f"❌ {convert_script} not found")
        return False
    
    # Check for Python dependencies
    try:
        import torch
        import transformers
        logger.info("✅ PyTorch and Transformers available")
    except ImportError:
        logger.info("📦 Installing required Python packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "transformers", "sentencepiece", "protobuf"], check=True)
    
    return True

def download_gemma_model(model_name, output_dir):
    """Download Gemma model from Hugging Face"""
    logger.info(f"📥 Downloading {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use Hugging Face CLI or Python API
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=output_dir)
        tokenizer.save_pretrained(f"{output_dir}/hf_model")
        
        logger.info(f"   Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(f"{output_dir}/hf_model")
        
        logger.info(f"✅ Model downloaded to {output_dir}/hf_model")
        return f"{output_dir}/hf_model"
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("💡 Alternative: Use Hugging Face CLI")
        logger.info(f"   huggingface-cli download {model_name} --local-dir {output_dir}/hf_model")
        return None

def convert_to_gguf(model_path, output_path, quantization="Q4_K_M"):
    """Convert HF model to GGUF format"""
    logger.info(f"🔄 Converting to GGUF format...")
    
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    output_file = f"{output_path}/gemma-4b-it.gguf"
    
    # Run conversion
    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile", output_file,
        "--outtype", quantization
    ]
    
    logger.info(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Converted to {output_file}")
            return output_file
        else:
            logger.error(f"❌ Conversion failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"❌ Conversion error: {e}")
        return None

def quantize_model(input_file, output_file, quantization="Q4_K_M"):
    """Quantize GGUF model for optimal NPU performance"""
    logger.info(f"⚡ Quantizing model to {quantization}...")
    
    quantize_exe = "llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_exe):
        logger.error(f"❌ {quantize_exe} not found - build llama.cpp first")
        return None
    
    cmd = [quantize_exe, input_file, output_file, quantization]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Quantized to {output_file}")
            return output_file
        else:
            logger.error(f"❌ Quantization failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"❌ Quantization error: {e}")
        return None

def main():
    logger.info("🦄 Gemma Model Preparation for NPU+iGPU Testing")
    logger.info("=" * 60)
    
    if not check_dependencies():
        return
    
    # Model options
    models = {
        "1": {
            "name": "google/gemma-2b-it",
            "size": "2B",
            "desc": "Smallest, good for testing"
        },
        "2": {
            "name": "google/gemma-7b-it",
            "size": "7B",
            "desc": "Standard size, balanced performance"
        },
        "3": {
            "name": "google/gemma-2-9b-it",
            "size": "9B",
            "desc": "Gemma 2 series, newer architecture"
        },
        "4": {
            "name": "google/gemma-2-27b-it",
            "size": "27B",
            "desc": "Largest, ultimate NPU+iGPU test"
        }
    }
    
    logger.info("Available models:")
    for key, model in models.items():
        logger.info(f"   {key}. {model['size']} - {model['desc']} ({model['name']})")
    
    # For automated testing, use Gemma 2B
    choice = "1"
    logger.info(f"\n📋 Selected: Gemma {models[choice]['size']} for NPU testing")
    
    model_name = models[choice]["name"]
    model_size = models[choice]["size"]
    output_dir = f"models/gemma-{model_size.lower()}"
    
    # Step 1: Download model
    logger.info(f"\n🚀 Step 1: Download {model_name}")
    model_path = download_gemma_model(model_name, output_dir)
    
    if not model_path:
        logger.info("\n💡 Manual download instructions:")
        logger.info("1. Install Hugging Face CLI: pip install huggingface-hub")
        logger.info("2. Login: huggingface-cli login")
        logger.info(f"3. Download: huggingface-cli download {model_name} --local-dir {output_dir}/hf_model")
        logger.info("4. Re-run this script")
        return
    
    # Step 2: Convert to GGUF
    logger.info(f"\n🚀 Step 2: Convert to GGUF")
    gguf_file = convert_to_gguf(model_path, output_dir)
    
    if not gguf_file:
        logger.info("\n💡 Manual conversion:")
        logger.info(f"python3 llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {output_dir}/gemma.gguf")
        return
    
    # Step 3: Quantize for NPU
    logger.info(f"\n🚀 Step 3: Quantize for NPU optimization")
    quantized_file = f"{output_dir}/gemma-{model_size.lower()}-q4_k_m.gguf"
    final_model = quantize_model(gguf_file, quantized_file)
    
    if final_model:
        logger.info("\n" + "=" * 60)
        logger.info(f"🎉 SUCCESS! Gemma {model_size} ready for NPU+iGPU testing")
        logger.info(f"📁 Model location: {final_model}")
        logger.info(f"📊 Model size: {os.path.getsize(final_model) / (1024**3):.2f} GB")
        logger.info("\n🚀 Test with NPU+iGPU acceleration:")
        logger.info(f"   ./llama.cpp/build/bin/llama-cli -m {final_model} -p \"Hello\" -n 50 --npu-attention --gpu-layers 999")
        logger.info("\n🧪 Benchmark performance:")
        logger.info(f"   python3 benchmark_npu_igpu_gemma.py --model {final_model}")
    else:
        logger.error("❌ Model preparation failed")

if __name__ == "__main__":
    main()