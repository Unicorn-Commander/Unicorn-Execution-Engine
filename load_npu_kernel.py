#!/usr/bin/env python3
"""
Load compiled NPU kernel for execution
"""

import os
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_best_kernel(seq_length: int, quantization: str = "int8") -> str:
    """Find the best matching NPU kernel"""
    
    kernel_dir = Path("/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels")
    
    # Check for exact match
    exact_match = kernel_dir / f"attention_{seq_length}_{quantization}.bin"
    if exact_match.exists():
        logger.info(f"✅ Found exact match: {exact_match.name}")
        return str(exact_match)
        
    # Check for flash attention
    flash_kernel = kernel_dir / "gemma-3n-e4b-attention" / "flash_attention_kernel.bin"
    if flash_kernel.exists():
        logger.info(f"✅ Found flash attention kernel: {flash_kernel.name}")
        return str(flash_kernel)
        
    # Find closest size
    available_sizes = []
    for kernel in kernel_dir.glob(f"attention_*_{quantization}.bin"):
        try:
            size = int(kernel.stem.split('_')[1])
            available_sizes.append((size, kernel))
        except:
            pass
            
    if available_sizes:
        # Sort by distance to requested size
        available_sizes.sort(key=lambda x: abs(x[0] - seq_length))
        best_kernel = available_sizes[0][1]
        logger.info(f"✅ Found closest match: {best_kernel.name} (size {available_sizes[0][0]})")
        return str(best_kernel)
        
    logger.error("❌ No suitable NPU kernel found")
    return None

def load_kernel_config() -> dict:
    """Load kernel configuration"""
    
    config_path = Path("/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels/gemma-3n-e4b-attention/kernel_configs.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info("✅ Loaded kernel configuration")
            return config
    else:
        # Default config
        return {
            "max_seq_length": 2048,
            "num_heads": 32,
            "head_dim": 168,
            "quantization": "int8",
            "flash_attention": True
        }

def setup_npu_kernel_path():
    """Setup NPU kernel path in environment"""
    
    kernel_path = find_best_kernel(256, "int8")
    
    if kernel_path:
        os.environ['NPU_KERNEL_PATH'] = kernel_path
        logger.info(f"✅ NPU kernel path set: {kernel_path}")
        
        # Also set config
        config = load_kernel_config()
        os.environ['NPU_CONFIG'] = json.dumps(config)
        
        return True
    return False

def test_kernel_loading():
    """Test loading NPU kernel"""
    
    logger.info("🧪 Testing NPU Kernel Loading...")
    
    # Find kernels for different sizes
    test_sizes = [256, 512, 1024, 2048]
    
    for size in test_sizes:
        logger.info(f"\nLooking for seq_length={size}:")
        
        for quant in ["int8", "int4"]:
            kernel = find_best_kernel(size, quant)
            if kernel:
                file_size = os.path.getsize(kernel) / 1024
                logger.info(f"   {quant}: {os.path.basename(kernel)} ({file_size:.1f} KB)")
                
    # Check performance metrics
    metrics_path = Path("/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels/gemma-3n-e4b-attention/performance_metrics.json")
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            logger.info("\n📊 NPU Kernel Performance Metrics:")
            for key, value in metrics.items():
                logger.info(f"   {key}: {value}")

def main():
    """Setup NPU kernel loading"""
    
    logger.info("🚀 NPU Kernel Setup")
    logger.info("=" * 50)
    
    # Test kernel discovery
    test_kernel_loading()
    
    # Setup default kernel
    if setup_npu_kernel_path():
        logger.info("\n✅ NPU kernel ready for execution!")
        logger.info(f"   Kernel: {os.environ.get('NPU_KERNEL_PATH')}")
        
        config = json.loads(os.environ.get('NPU_CONFIG', '{}'))
        logger.info("   Configuration:")
        for key, value in config.items():
            logger.info(f"     {key}: {value}")
    else:
        logger.error("\n❌ Failed to setup NPU kernel")

if __name__ == "__main__":
    main()