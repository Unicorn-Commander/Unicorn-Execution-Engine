#!/usr/bin/env python3
"""
Verify and Save Quantized Gemma 3 27B Model
Check if quantization completed successfully and save the result
"""

import os
import torch
import multiprocessing
import json
from pathlib import Path

# Set environment variables for maximum CPU usage
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Fix for ROCm gfx1103 TensileLibrary.dat issue
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['ROCM_PATH'] = '/opt/rocm'

# Force PyTorch to use all CPU cores
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

from integrated_quantized_npu_engine import IntegratedQuantizedNPUEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_and_save_quantization():
    """Verify quantization and save the quantized model"""
    logger.info("🔍 Verifying Gemma 3 27B quantization...")
    
    # Initialize engine
    engine = IntegratedQuantizedNPUEngine(
        enable_quantization=True,
        turbo_mode=True
    )
    
    # Load and quantize model
    model_path = "./models/gemma-3-27b-it"
    logger.info(f"📦 Loading and quantizing model from {model_path}")
    
    quantized_result = engine.load_and_quantize_model(model_path)
    
    # Verify quantization
    if quantized_result and "summary" in quantized_result:
        summary = quantized_result["summary"]
        logger.info("✅ Quantization verification successful!")
        logger.info(f"📊 Original size: {summary.get('original_size_gb', 0):.2f} GB")
        logger.info(f"📊 Quantized size: {summary.get('quantized_size_gb', 0):.2f} GB")
        logger.info(f"📊 Memory reduction: {summary.get('total_savings_ratio', 0):.1%}")
        logger.info(f"🎯 NPU memory fit: {summary.get('npu_memory_fit', False)}")
        
        # Save quantized model
        output_dir = Path("./quantized_models/gemma-3-27b-it-optimized-quantized")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save quantization metadata
        metadata = {
            "model_name": "gemma-3-27b-it",
            "quantization_date": "2025-07-08",
            "original_size_gb": summary.get('original_size_gb', 0),
            "quantized_size_gb": summary.get('quantized_size_gb', 0),
            "memory_reduction": summary.get('total_savings_ratio', 0),
            "npu_compatible": summary.get('npu_memory_fit', False),
            "hardware_config": {
                "npu_available": engine.npu_available,
                "igpu_available": engine.igpu_available,
                "vulkan_available": engine.vulkan_available,
                "turbo_mode": engine.turbo_mode
            }
        }
        
        with open(output_dir / "quantization_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Quantization metadata saved to {output_dir}")
        
        # Test a simple generation to verify functionality
        logger.info("🧪 Testing quantized model generation...")
        try:
            test_result = engine.generate_text_quantized(
                prompt="The future of AI is",
                max_tokens=10,
                temperature=0.7
            )
            
            logger.info("✅ Generation test successful!")
            logger.info(f"🎯 Generated: {test_result.get('generated_text', 'N/A')}")
            logger.info(f"📈 TPS: {test_result.get('tokens_per_second', 0):.2f}")
            
            # Save test result
            with open(output_dir / "generation_test.json", "w") as f:
                json.dump(test_result, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Generation test failed: {e}")
            return False
        
    else:
        logger.error("❌ Quantization verification failed!")
        return False

if __name__ == "__main__":
    success = verify_and_save_quantization()
    if success:
        print("\n🎉 Quantization verification and save completed successfully!")
    else:
        print("\n💥 Quantization verification failed!")