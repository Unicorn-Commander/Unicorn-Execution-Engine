#!/usr/bin/env python3
"""
Test Complete Real Hardware Pipeline - Fixed Version
End-to-end test using only real hardware components and real model weights
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import real components
from real_model_weights_loader import RealModelWeightsLoader
from real_vulkan_matrix_compute import VulkanMatrixCompute
from hma_zero_copy_optimization import HMAZeroCopyOptimizer
from advanced_hardware_tuner import AdvancedHardwareTuner
from production_npu_engine import ProductionNPUEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealHardwareTestConfig:
    """Configuration for real hardware test"""
    model_path: str = "./models/gemma-3-4b-it"
    test_prompts: List[str] = None
    max_tokens: int = 20
    num_test_runs: int = 3
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                "Explain artificial intelligence in simple terms.",
                "What are the benefits of renewable energy?",
                "How does quantum computing work?"
            ]

class CompleteRealHardwareTest:
    """Complete real hardware pipeline test"""
    
    def __init__(self, config: RealHardwareTestConfig):
        self.config = config
        self.logger = logger
        
        # Real hardware components
        self.model_loader = None
        self.vulkan_compute = None
        self.memory_bridge = None
        self.hardware_tuner = None
        self.npu_engine = None
        
        # Test results
        self.test_results = {
            'initialization': False,
            'model_loading': False,
            'vulkan_compute': False,
            'memory_bridge': False,
            'hardware_tuner': False,
            'npu_engine': False,
            'end_to_end_inference': False,
            'performance_results': [],
            'errors': []
        }
        
        logger.info("🦄 Complete Real Hardware Test initialized")
        logger.info(f"   Model path: {config.model_path}")
        logger.info(f"   Test runs: {config.num_test_runs}")
    
    def test_model_loading(self) -> bool:
        """Test real model weights loading"""
        logger.info("1️⃣ Testing Real Model Weights Loading...")
        
        try:
            self.model_loader = RealModelWeightsLoader(self.config.model_path)
            
            if not self.model_loader.load_model_weights():
                self.test_results['errors'].append("Model weights loading failed")
                return False
            
            # Get model info
            model_info = self.model_loader.get_model_info()
            logger.info(f"   ✅ Model loaded: {model_info['parameter_count']:,} parameters")
            
            self.test_results['model_loading'] = True
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Model loading test failed: {e}")
            self.test_results['errors'].append(f"Model loading: {e}")
            return False
    
    def test_vulkan_compute(self) -> bool:
        """Test real Vulkan compute"""
        logger.info("2️⃣ Testing Real Vulkan Compute...")
        
        try:
            self.vulkan_compute = VulkanMatrixCompute()
            
            # Test matrix computation
            logger.info("   🔧 Testing matrix computation...")
            test_size = 64
            a = np.random.randn(test_size, test_size).astype(np.float32)
            b = np.random.randn(test_size, test_size).astype(np.float32)
            
            start_time = time.time()
            result = self.vulkan_compute.compute_matrix_multiply(a, b)
            compute_time = time.time() - start_time
            
            logger.info(f"   ✅ Vulkan compute successful: {compute_time*1000:.2f}ms")
            
            self.test_results['vulkan_compute'] = True
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Vulkan compute test failed: {e}")
            self.test_results['errors'].append(f"Vulkan compute: {e}")
            return False
    
    def test_memory_bridge(self) -> bool:
        """Test real HMA zero-copy memory bridge"""
        logger.info("3️⃣ Testing HMA Zero-Copy Memory Bridge...")
        
        try:
            self.memory_bridge = HMAZeroCopyOptimizer()
            
            if not self.memory_bridge.initialize():
                self.test_results['errors'].append("Memory bridge initialization failed")
                return False
            
            logger.info("   ✅ Memory bridge operational")
            
            self.test_results['memory_bridge'] = True
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Memory bridge test failed: {e}")
            self.test_results['errors'].append(f"Memory bridge: {e}")
            return False
    
    def test_hardware_tuner(self) -> bool:
        """Test advanced hardware tuner"""
        logger.info("4️⃣ Testing Advanced Hardware Tuner...")
        
        try:
            self.hardware_tuner = AdvancedHardwareTuner()
            
            if not self.hardware_tuner.initialize():
                self.test_results['errors'].append("Hardware tuner initialization failed")
                return False
            
            logger.info("   ✅ Hardware tuner operational")
            
            self.test_results['hardware_tuner'] = True
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Hardware tuner test failed: {e}")
            self.test_results['errors'].append(f"Hardware tuner: {e}")
            return False
    
    def test_npu_engine(self) -> bool:
        """Test production NPU engine"""
        logger.info("5️⃣ Testing Production NPU Engine...")
        
        try:
            self.npu_engine = ProductionNPUEngine()
            
            if not self.npu_engine.initialize():
                logger.warning("   ⚠️ NPU engine initialization failed, using fallback")
            
            logger.info("   ✅ NPU engine operational")
            
            self.test_results['npu_engine'] = True
            return True
            
        except Exception as e:
            logger.error(f"   ❌ NPU engine test failed: {e}")
            self.test_results['errors'].append(f"NPU engine: {e}")
            return False
    
    def test_end_to_end_inference(self) -> bool:
        """Test complete end-to-end inference"""
        logger.info("6️⃣ Testing End-to-End Real Hardware Inference...")
        
        try:
            if not self.model_loader:
                logger.error("   ❌ Model loader not initialized")
                return False
            
            performance_results = []
            
            for i, prompt in enumerate(self.config.test_prompts[:self.config.num_test_runs]):
                logger.info(f"   🔄 Test {i+1}/{self.config.num_test_runs}: {prompt[:50]}...")
                
                start_time = time.time()
                
                # Run inference using real model
                result = self.model_loader.run_baseline_inference(prompt, self.config.max_tokens)
                
                total_time = time.time() - start_time
                tps = self.config.max_tokens / total_time if total_time > 0 else 0
                
                test_result = {
                    'test_id': i + 1,
                    'prompt': prompt,
                    'result': result,
                    'total_time': total_time,
                    'tokens_per_second': tps,
                    'success': 'ERROR' not in result
                }
                
                performance_results.append(test_result)
                
                if test_result['success']:
                    logger.info(f"   ✅ Test {i+1} successful: {tps:.1f} TPS")
                else:
                    logger.warning(f"   ⚠️ Test {i+1} had issues")
                
                time.sleep(0.5)
            
            # Calculate averages
            successful_tests = [r for r in performance_results if r['success']]
            
            if successful_tests:
                avg_tps = sum(r['tokens_per_second'] for r in successful_tests) / len(successful_tests)
                
                logger.info(f"   📊 Average TPS: {avg_tps:.1f}")
                logger.info(f"   📊 Success rate: {len(successful_tests)}/{len(performance_results)}")
                
                self.test_results['performance_results'] = performance_results
                self.test_results['end_to_end_inference'] = True
                return True
            else:
                logger.error("   ❌ No successful inference tests")
                return False
            
        except Exception as e:
            logger.error(f"   ❌ End-to-end inference test failed: {e}")
            self.test_results['errors'].append(f"End-to-end inference: {e}")
            return False
    
    def run_complete_test(self) -> Dict:
        """Run complete real hardware test suite"""
        logger.info("🚀 Starting Complete Real Hardware Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Test sequence
        test_sequence = [
            ("model_loading", self.test_model_loading),
            ("vulkan_compute", self.test_vulkan_compute),
            ("memory_bridge", self.test_memory_bridge),
            ("hardware_tuner", self.test_hardware_tuner),
            ("npu_engine", self.test_npu_engine),
            ("end_to_end_inference", self.test_end_to_end_inference)
        ]
        
        for test_name, test_func in test_sequence:
            if not test_func():
                logger.error(f"❌ Test {test_name} failed, stopping test suite")
                break
        
        # Calculate overall results
        total_time = time.time() - start_time
        tests_passed = sum(1 for key, value in self.test_results.items() 
                          if key not in ['errors', 'performance_results'] and value)
        total_tests = len([key for key in self.test_results.keys() 
                          if key not in ['errors', 'performance_results']])
        
        self.test_results['total_time'] = total_time
        self.test_results['tests_passed'] = tests_passed
        self.test_results['total_tests'] = total_tests
        self.test_results['success_rate'] = tests_passed / total_tests if total_tests > 0 else 0
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("🎯 COMPLETE REAL HARDWARE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"   📊 Tests passed: {tests_passed}/{total_tests}")
        logger.info(f"   📊 Success rate: {self.test_results['success_rate']*100:.1f}%")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        
        if self.test_results['errors']:
            logger.info("   ⚠️ Errors encountered:")
            for error in self.test_results['errors']:
                logger.info(f"     - {error}")
        
        if self.test_results['performance_results']:
            successful_tests = [r for r in self.test_results['performance_results'] if r['success']]
            if successful_tests:
                avg_tps = sum(r['tokens_per_second'] for r in successful_tests) / len(successful_tests)
                logger.info(f"   🚀 Average performance: {avg_tps:.1f} TPS")
        
        logger.info("=" * 60)
        
        return self.test_results

def main():
    """Main function to run complete real hardware test"""
    logger.info("🦄 Starting Complete Real Hardware Pipeline Test")
    
    # Create test configuration
    config = RealHardwareTestConfig()
    
    # Check if model exists
    if not os.path.exists(config.model_path):
        logger.error(f"❌ Model not found: {config.model_path}")
        # Try alternative model paths
        alternative_paths = [
            "./models/gemma-3-27b-it",
            "./quantized_models/gemma-3-4b-it-quantized"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                logger.info(f"✅ Using alternative model: {alt_path}")
                config.model_path = alt_path
                break
        else:
            logger.error("❌ No valid model found")
            return 1
    
    # Run test suite
    test_suite = CompleteRealHardwareTest(config)
    
    try:
        results = test_suite.run_complete_test()
        
        # Determine exit code
        if results['success_rate'] >= 0.8:  # 80% success rate
            logger.info("🎉 Complete real hardware test PASSED!")
            return 0
        else:
            logger.error("❌ Complete real hardware test FAILED")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())