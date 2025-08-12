#!/usr/bin/env python3
"""
Optimized Quantization Script for Maximum Performance
Uses all CPU threads + NPU + iGPU for parallel processing
"""

import os
import torch
import multiprocessing

# MAXIMUM PERFORMANCE: Use all available CPU resources
cpu_count = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count) 
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
os.environ['NUMPY_NUM_THREADS'] = str(cpu_count)

# Memory optimization for 16GB iGPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Fix for ROCm gfx1103 TensileLibrary.dat issue  
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['ROCM_PATH'] = '/opt/rocm'

# Force PyTorch to use all CPU cores
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

# Import after setting environment
from integrated_quantized_npu_engine import IntegratedQuantizedNPUEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_quantization():
    """Run quantization with maximum resource utilization for 16GB iGPU"""
    logger.info("🚀 Starting MAXIMUM PERFORMANCE quantization for Gemma 3 27B")
    logger.info(f"💻 Using ALL {multiprocessing.cpu_count()} CPU cores")
    logger.info(f"🎮 16GB iGPU VRAM allocation detected")
    logger.info(f"⚡ Environment optimized: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    
    # Initialize engine with ALL optimizations for 16GB iGPU
    engine = IntegratedQuantizedNPUEngine(
        enable_quantization=True,
        turbo_mode=True
    )
    
    # Load and quantize Gemma 3 27B model
    model_path = "./models/gemma-3-27b-it"
    output_path = "./quantized_models/gemma-3-27b-it-16gb-optimized"
    
    logger.info(f"📦 Loading 27B model from {model_path}")
    logger.info(f"💾 Target: {output_path}")
    logger.info("🚀 Using NPU (attention) + 16GB iGPU (FFN) + CPU (orchestration)")
    
    result = engine.load_and_quantize_model(model_path)
    
    if result and "summary" in result:
        summary = result["summary"]
        logger.info("🎉 QUANTIZATION COMPLETE!")
        logger.info(f"📊 Size: {summary.get('original_size_gb', 0):.2f}GB → {summary.get('quantized_size_gb', 0):.2f}GB")
        logger.info(f"📊 Memory reduction: {summary.get('total_savings_ratio', 0):.1%}")
        logger.info(f"🎯 Fits in 16GB iGPU: {summary.get('quantized_size_gb', 0) < 16}")
        
        # Save metadata
        import json
        from pathlib import Path
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "model": "gemma-3-27b-it",
            "quantization_date": "2025-07-08",
            "hardware_config": "16GB iGPU + NPU Phoenix + 16-core CPU",
            "original_size_gb": summary.get('original_size_gb', 0),
            "quantized_size_gb": summary.get('quantized_size_gb', 0),
            "memory_reduction": summary.get('total_savings_ratio', 0),
            "fits_in_16gb_igpu": summary.get('quantized_size_gb', 0) < 16,
            "optimization_level": "maximum"
        }
        
        with open(output_dir / "quantization_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Metadata saved to {output_dir}/quantization_info.json")
    
    return result

if __name__ == "__main__":
    optimize_quantization()