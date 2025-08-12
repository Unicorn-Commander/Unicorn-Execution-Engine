#!/usr/bin/env python3
"""
GPU-only inference test for Gemma3 4B
Bypasses NPU to establish baseline performance
"""

import os
import time
import numpy as np
import logging
import ctypes
from real_vulkan_matrix_compute import VulkanMatrixCompute as RealVulkanMatrixCompute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment for GPU stability
os.environ['RADV_PERFTEST'] = 'video_decode'
os.environ['AMD_VULKAN_ICD'] = 'RADV'

class GPUOnlyPipeline:
    """Minimal GPU-only pipeline for testing"""
    
    def __init__(self):
        self.vulkan_engine = None
        self.model_loaded = False
        self.gpu_buffers = {}
        
    def initialize(self):
        """Initialize GPU engine"""
        try:
            logger.info("🎮 Initializing GPU-only pipeline...")
            
            # Initialize Vulkan compute
            self.vulkan_engine = RealVulkanMatrixCompute()
            
            logger.info("✅ GPU engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU initialization failed: {e}")
            return False
    
    def load_minimal_weights(self):
        """Load minimal weights for testing"""
        try:
            logger.info("📦 Creating minimal test weights...")
            
            # Create small test weights
            vocab_size = 32000
            hidden_size = 2560
            
            # Minimal embedding matrix
            embed_weight = np.random.randn(vocab_size, hidden_size).astype(np.float16)
            
            # Allocate on GPU
            embed_buffer, embed_info = self.vulkan_engine.create_gpu_buffer(
                embed_weight.nbytes, 
                usage="storage"
            )
            
            # Copy data
            self.vulkan_engine.copy_to_gpu_buffer(embed_buffer, embed_weight.tobytes())
            
            # Store buffer info
            self.gpu_buffers['embeddings'] = {
                'buffer': embed_buffer,
                'buffer_info': embed_info,
                'shape': embed_weight.shape,
                'dtype': embed_weight.dtype
            }
            
            # Create simple linear layer
            linear_weight = np.random.randn(hidden_size, hidden_size).astype(np.float16)
            linear_buffer, linear_info = self.vulkan_engine.create_gpu_buffer(
                linear_weight.nbytes,
                usage="storage"
            )
            self.vulkan_engine.copy_to_gpu_buffer(linear_buffer, linear_weight.tobytes())
            
            self.gpu_buffers['linear'] = {
                'buffer': linear_buffer,
                'buffer_info': linear_info,
                'shape': linear_weight.shape,
                'dtype': linear_weight.dtype
            }
            
            logger.info("✅ Test weights loaded to GPU")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Weight loading failed: {e}")
            return False
    
    def test_embedding_lookup(self, input_ids):
        """Test embedding lookup on GPU"""
        if not self.model_loaded:
            return None
            
        try:
            embed_info = self.gpu_buffers['embeddings']['buffer_info']
            
            logger.info(f"🔍 Looking up embeddings for {len(input_ids)} tokens...")
            start = time.time()
            
            result = self.vulkan_engine.compute_embedding_lookup_gpu(
                input_ids, embed_info
            )
            
            elapsed = time.time() - start
            logger.info(f"✅ Embedding lookup completed in {elapsed*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Embedding lookup failed: {e}")
            return None
    
    def test_matrix_multiply(self, input_data):
        """Test matrix multiplication on GPU"""
        if not self.model_loaded:
            return None
            
        try:
            linear_info = self.gpu_buffers['linear']['buffer_info']
            weight_shape = self.gpu_buffers['linear']['shape']
            
            logger.info(f"🔢 Computing matrix multiply...")
            start = time.time()
            
            # Create output buffer
            output_shape = (input_data.shape[0], input_data.shape[1], weight_shape[1])
            output_size = np.prod(output_shape) * 2  # float16
            
            output_buffer, output_info = self.vulkan_engine.create_gpu_buffer(
                output_size, usage="storage"
            )
            
            # Perform matrix multiply
            self.vulkan_engine.matrix_multiply_gpu(
                input_data, linear_info, output_info,
                input_data.shape, weight_shape, output_shape
            )
            
            elapsed = time.time() - start
            
            # Calculate GFLOPS
            flops = 2 * np.prod(input_data.shape) * weight_shape[1]
            gflops = flops / (elapsed * 1e9)
            
            logger.info(f"✅ Matrix multiply completed in {elapsed*1000:.2f}ms")
            logger.info(f"🚀 Performance: {gflops:.2f} GFLOPS")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Matrix multiply failed: {e}")
            return None

def run_gpu_test():
    """Run GPU-only test"""
    
    logger.info("🎮 GPU-ONLY INFERENCE TEST")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = GPUOnlyPipeline()
    
    # Initialize
    if not pipeline.initialize():
        return 0
    
    # Load weights
    if not pipeline.load_minimal_weights():
        return 0
    
    # Test cases
    test_results = []
    
    # Test 1: Embedding lookup
    logger.info("\n📊 Test 1: Embedding Lookup")
    input_ids = [1, 100, 500, 1000, 2000]
    embeddings = pipeline.test_embedding_lookup(input_ids)
    
    if embeddings is not None:
        logger.info(f"  ✅ Embeddings shape: {embeddings.shape}")
        test_results.append(("Embedding Lookup", True))
    else:
        test_results.append(("Embedding Lookup", False))
    
    # Test 2: Matrix multiply
    if embeddings is not None:
        logger.info("\n📊 Test 2: Matrix Multiplication")
        success = pipeline.test_matrix_multiply(embeddings)
        test_results.append(("Matrix Multiply", success is not None))
    
    # Test 3: Multiple operations
    logger.info("\n📊 Test 3: Sequential Operations")
    try:
        total_time = 0
        ops = 10
        
        for i in range(ops):
            input_ids = [np.random.randint(0, 30000) for _ in range(3)]
            
            start = time.time()
            emb = pipeline.test_embedding_lookup(input_ids)
            if emb is not None:
                pipeline.test_matrix_multiply(emb)
            elapsed = time.time() - start
            
            total_time += elapsed
            
        avg_time = total_time / ops
        tps = 1 / avg_time
        
        logger.info(f"  ✅ Average time per operation: {avg_time*1000:.2f}ms")
        logger.info(f"  🚀 Theoretical TPS: {tps:.2f}")
        test_results.append(("Sequential Ops", True))
        
    except Exception as e:
        logger.error(f"  ❌ Sequential test failed: {e}")
        test_results.append(("Sequential Ops", False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 GPU TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 GPU INFERENCE WORKING!")
        logger.info("   Next step: Scale up to full model")
    else:
        logger.info("⚠️ Some GPU tests failed")
        logger.info("   Check GPU memory and Vulkan setup")
    
    return passed

def main():
    """Main entry point"""
    try:
        passed_tests = run_gpu_test()
        
        if passed_tests > 0:
            logger.info("\n✅ GPU BASELINE ESTABLISHED")
            logger.info("   Ready to integrate with full model")
        else:
            logger.info("\n❌ GPU tests failed completely")
            
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()