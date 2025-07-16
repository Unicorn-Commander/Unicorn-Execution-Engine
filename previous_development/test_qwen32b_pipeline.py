#!/usr/bin/env python3
"""
Qwen 2.5 32B Pipeline Test
End-to-end test of the NPU+iGPU optimized pipeline
"""

import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen32b_components():
    """Test all Qwen 32B components"""
    
    logger.info("🦄 Testing Qwen 2.5 32B Unicorn Pipeline Components")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Memory Allocator
    try:
        logger.info("1️⃣ Testing Memory Allocator...")
        from qwen32b_npu_igpu_memory_allocator import Qwen32BMemoryAllocator
        
        allocator = Qwen32BMemoryAllocator()
        memory_map = allocator.create_memory_map()
        
        if memory_map and len(memory_map["layers"]) > 0:
            results["memory_allocator"] = "✅ PASSED"
            logger.info(f"   ✅ Allocated {len(memory_map['layers'])} layers")
        else:
            results["memory_allocator"] = "❌ FAILED"
    except Exception as e:
        results["memory_allocator"] = f"❌ FAILED: {e}"
    
    # Test 2: HMA Bridge
    try:
        logger.info("2️⃣ Testing HMA Memory Bridge...")
        from qwen32b_hma_memory_bridge import Qwen32BHMAMemoryBridge
        
        bridge = Qwen32BHMAMemoryBridge()
        layout = bridge.create_qwen32b_memory_layout()
        
        if layout and "model_sharding" in layout:
            results["hma_bridge"] = "✅ PASSED"
            logger.info(f"   ✅ Created memory layout with device mapping")
        else:
            results["hma_bridge"] = "❌ FAILED"
    except Exception as e:
        results["hma_bridge"] = f"❌ FAILED: {e}"
    
    # Test 3: NPU Kernels
    try:
        logger.info("3️⃣ Testing NPU Attention Kernels...")
        from qwen32b_npu_attention_kernels import Qwen32BNPUAttentionKernel
        
        kernel_compiler = Qwen32BNPUAttentionKernel()
        compiled_kernels = kernel_compiler.compile_kernels()
        
        if compiled_kernels and len(compiled_kernels) > 0:
            results["npu_kernels"] = "✅ PASSED"
            logger.info(f"   ✅ Compiled {len(compiled_kernels)} NPU kernels")
        else:
            results["npu_kernels"] = "❌ FAILED"
    except Exception as e:
        results["npu_kernels"] = f"❌ FAILED: {e}"
    
    # Test 4: Vulkan Shaders
    try:
        logger.info("4️⃣ Testing Vulkan FFN Shaders...")
        from qwen32b_vulkan_ffn_shaders import Qwen32BVulkanFFNShaders
        
        shader_compiler = Qwen32BVulkanFFNShaders()
        compiled_shaders = shader_compiler.compile_shaders()
        
        if compiled_shaders and len(compiled_shaders) > 0:
            results["vulkan_shaders"] = "✅ PASSED"
            logger.info(f"   ✅ Compiled {len(compiled_shaders)} Vulkan shaders")
        else:
            results["vulkan_shaders"] = "❌ FAILED"
    except Exception as e:
        results["vulkan_shaders"] = f"❌ FAILED: {e}"
    
    # Test 5: Unicorn Loader
    try:
        logger.info("5️⃣ Testing Unicorn Loader...")
        from qwen32b_unicorn_loader import Qwen32BUnicornLoader
        
        loader = Qwen32BUnicornLoader()
        architecture = loader.analyze_model_architecture()
        
        if architecture and "num_layers" in architecture:
            shards = loader.create_sharding_strategy()
            contexts = loader.initialize_hardware_contexts()
            
            results["unicorn_loader"] = "✅ PASSED"
            logger.info(f"   ✅ Created {len(shards)} shards with {len(contexts)} contexts")
        else:
            results["unicorn_loader"] = "❌ FAILED"
    except Exception as e:
        results["unicorn_loader"] = f"❌ FAILED: {e}"
    
    return results

def test_model_exists():
    """Test if Qwen 32B model exists"""
    
    model_path = Path("./models/qwen2.5-32b-instruct")
    
    if model_path.exists():
        config_file = model_path / "config.json"
        if config_file.exists():
            logger.info(f"✅ Qwen 32B model found at {model_path}")
            return True
        else:
            logger.warning(f"⚠️ Model directory exists but config.json missing")
            return False
    else:
        logger.warning(f"⚠️ Qwen 32B model not found at {model_path}")
        return False

def create_performance_summary():
    """Create performance targets summary"""
    
    logger.info("🎯 Qwen 2.5 32B Performance Targets")
    logger.info("-" * 40)
    
    targets = {
        "NPU Phoenix (Attention)": {
            "layers": "0-20 (21 layers)",
            "precision": "INT8 symmetric",
            "memory": "2GB SRAM",
            "target_tps": "50-100 TPS"
        },
        "Radeon 780M (FFN)": {
            "layers": "21-44 (24 layers)", 
            "precision": "INT4 grouped",
            "memory": "16GB DDR5",
            "target_tps": "30-80 TPS"
        },
        "System Memory (Mixed)": {
            "layers": "45-63 (19 layers)",
            "precision": "FP16",
            "memory": "80GB DDR5",
            "target_tps": "10-30 TPS"
        }
    }
    
    for device, specs in targets.items():
        logger.info(f"{device}:")
        for key, value in specs.items():
            logger.info(f"  {key}: {value}")
        logger.info("")
    
    logger.info("🎯 Overall Target: 90-210 TPS (3-7x speedup vs CPU)")
    logger.info("💾 Memory Reduction: 60-70% (32B params → 10-12GB)")

def main():
    """Main test function"""
    
    logger.info("🦄 Qwen 2.5 32B NPU+iGPU Pipeline Test")
    logger.info("=" * 60)
    
    # Check model availability
    model_exists = test_model_exists()
    
    # Test components
    results = test_qwen32b_components()
    
    # Performance summary
    create_performance_summary()
    
    # Final results
    logger.info("=" * 60)
    logger.info("🎯 COMPONENT TEST RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for component, result in results.items():
        logger.info(f"{component:20}: {result}")
        if "✅ PASSED" in result:
            passed += 1
    
    logger.info(f"\n📊 Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if model_exists:
        logger.info("✅ Qwen 32B model available for quantization")
    else:
        logger.info("⚠️ Qwen 32B model not found - download required")
    
    logger.info("=" * 60)
    logger.info("🚀 NEXT STEPS:")
    logger.info("1. Run quantization: python qwen32b_unicorn_quantization_engine.py")
    logger.info("2. Start API server: python qwen32b_openai_api_server.py")
    logger.info("3. Test with OpenWebUI: http://localhost:8000/v1")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())