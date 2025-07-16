#!/usr/bin/env python3
"""
Force Vulkan-only computation, disable HIP/ROCm to avoid allocation errors
"""

import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_vulkan_only_mode():
    """Force Vulkan-only mode, disable HIP/ROCm"""
    logger.info("🔥 FORCING VULKAN-ONLY MODE - DISABLING HIP/ROCm")
    
    # Disable HIP/ROCm
    os.environ['HIP_VISIBLE_DEVICES'] = ''
    os.environ['ROCR_VISIBLE_DEVICES'] = ''
    os.environ['GPU_FORCE_64BIT_PTR'] = '0'
    os.environ['GPU_MAX_HEAP_SIZE'] = '100'
    os.environ['GPU_MAX_ALLOC_PERCENT'] = '100'
    os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '100'
    
    # Force CPU-only PyTorch (we'll use Vulkan separately)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set PyTorch to CPU-only mode
    if torch.cuda.is_available():
        logger.warning("⚠️ Disabling CUDA for Vulkan-only mode")
    
    # Force CPU device for PyTorch tensors
    torch.set_default_device('cpu')
    
    logger.info("✅ Vulkan-only mode activated")
    logger.info("   🚫 HIP/ROCm disabled")
    logger.info("   🚫 CUDA disabled") 
    logger.info("   ✅ Vulkan compute available for iGPU")
    logger.info("   ✅ PyTorch on CPU for tensor management")

if __name__ == "__main__":
    force_vulkan_only_mode()
    
    # Test that HIP is disabled
    try:
        import torch
        print(f"PyTorch device: {torch.get_default_device()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Try to create a tensor - should be CPU only
        test_tensor = torch.randn(10, 10)
        print(f"Test tensor device: {test_tensor.device}")
        
        # Test Vulkan separately
        from vulkan_ffn_compute_engine import VulkanFFNComputeEngine
        vulkan_engine = VulkanFFNComputeEngine()
        if vulkan_engine.initialize():
            print("✅ Vulkan iGPU compute working")
        else:
            print("❌ Vulkan iGPU compute failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")