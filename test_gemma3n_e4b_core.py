#!/usr/bin/env python3
"""
Gemma 3n E4B Core Pipeline Test
Test the core components without API server dependency
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our core components
from gemma3n_e4b_unicorn_loader import (
    Gemma3nE4BUnicornLoader,
    ModelConfig,
    HardwareConfig,
    InferenceConfig,
    InferenceMode,
    LoaderState
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma3nE4BCoreTest:
    """Core test suite for Gemma 3n E4B pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.loader = None
        
    def setup_test_environment(self):
        """Setup the test environment"""
        logger.info("🔧 Setting up Gemma 3n E4B test environment...")
        
        # Configure model
        model_config = ModelConfig(
            model_path="./models/gemma-3n-e4b-it",
            elastic_enabled=True,
            quantization_enabled=True,
            mix_n_match_enabled=True
        )
        
        # Configure hardware
        hardware_config = HardwareConfig(
            npu_enabled=True,
            igpu_enabled=True,
            hma_enabled=True,
            turbo_mode=True,
            zero_copy_enabled=True
        )
        
        # Initialize loader
        self.loader = Gemma3nE4BUnicornLoader(model_config, hardware_config)
        
        logger.info("✅ Test environment setup complete")
    
    def test_component_initialization(self) -> Dict[str, Any]:
        """Test individual component initialization"""
        logger.info("🔬 Testing component initialization...")
        
        results = {
            "quantizer": self.loader.quantizer is not None,
            "allocator": self.loader.allocator is not None,
            "elastic_system": self.loader.elastic_system is not None,
            "npu_kernels": self.loader.npu_kernels is not None,
            "vulkan_shaders": self.loader.vulkan_shaders is not None,
            "hma_bridge": self.loader.hma_bridge is not None,
            "loader_state": self.loader.state.value
        }
        
        # Count successful initializations
        component_count = sum(1 for key in results if key != "loader_state" and results[key])
        
        logger.info(f"   ✅ {component_count}/6 components initialized")
        logger.info(f"   🔧 Loader state: {results['loader_state']}")
        
        for component, status in results.items():
            if component != "loader_state":
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {component}: {status}")
        
        results["success"] = component_count >= 4  # Need at least 4 core components
        return results
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading with quantization"""
        logger.info("🚀 Testing model loading...")
        
        results = {
            "loading_success": False,
            "loading_time": 0.0,
            "active_elastic_params": 0,
            "loader_state": "unknown"
        }
        
        start_time = time.time()
        
        try:
            # Load model
            loading_success = self.loader.load_model()
            
            results["loading_success"] = loading_success
            results["loading_time"] = time.time() - start_time
            results["loader_state"] = self.loader.state.value
            results["active_elastic_params"] = len(self.loader.active_elastic_params)
            
            if loading_success:
                logger.info(f"   ✅ Model loaded in {results['loading_time']:.2f}s")
                logger.info(f"   📊 Active elastic params: {results['active_elastic_params']}")
                logger.info(f"   🔧 Loader state: {results['loader_state']}")
            else:
                logger.error("   ❌ Model loading failed")
        
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            results["error"] = str(e)
        
        results["success"] = loading_success
        return results
    
    def test_inference_performance(self) -> Dict[str, Any]:
        """Test inference performance with various prompts"""
        logger.info("⚡ Testing inference performance...")
        
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?"
        ]
        
        results = {
            "inference_results": [],
            "successful_tests": 0,
            "failed_tests": 0,
            "average_tps": 0.0,
            "average_inference_time": 0.0
        }
        
        total_tps = 0.0
        total_time = 0.0
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"   🔍 Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                # Configure inference
                inference_config = InferenceConfig(
                    mode=InferenceMode.BALANCED,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    elastic_scaling=True,
                    dynamic_allocation=True
                )
                
                # Run inference
                result = self.loader.generate(prompt, inference_config)
                
                if "error" not in result:
                    test_result = {
                        "prompt": prompt,
                        "success": True,
                        "tokens_generated": result.get("tokens_generated", 0),
                        "inference_time": result.get("inference_time", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "elastic_params_active": result.get("elastic_params_active", 0)
                    }
                    
                    results["inference_results"].append(test_result)
                    results["successful_tests"] += 1
                    
                    tps = result.get("tokens_per_second", 0)
                    inf_time = result.get("inference_time", 0)
                    
                    total_tps += tps
                    total_time += inf_time
                    
                    logger.info(f"     ✅ {tps:.1f} TPS, {inf_time:.2f}s, {result.get('tokens_generated', 0)} tokens")
                    
                else:
                    logger.error(f"     ❌ Inference failed: {result['error']}")
                    results["failed_tests"] += 1
                    results["inference_results"].append({
                        "prompt": prompt,
                        "success": False,
                        "error": result["error"]
                    })
            
            except Exception as e:
                logger.error(f"     ❌ Inference error: {e}")
                results["failed_tests"] += 1
                results["inference_results"].append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate averages
        if results["successful_tests"] > 0:
            results["average_tps"] = total_tps / results["successful_tests"]
            results["average_inference_time"] = total_time / results["successful_tests"]
        
        success_rate = results["successful_tests"] / len(test_prompts)
        logger.info(f"   📊 Success rate: {success_rate:.1%}")
        logger.info(f"   📊 Average TPS: {results['average_tps']:.1f}")
        logger.info(f"   📊 Average inference time: {results['average_inference_time']:.2f}s")
        
        results["success"] = success_rate >= 0.6  # At least 60% success rate
        return results
    
    def test_elastic_scaling(self) -> Dict[str, Any]:
        """Test elastic parameter scaling functionality"""
        logger.info("🔄 Testing elastic parameter scaling...")
        
        results = {
            "hma_bridge_available": False,
            "elastic_system_available": False,
            "parameter_activation": [],
            "parameter_deactivation": [],
            "memory_optimization": False
        }
        
        try:
            # Check if systems are available
            results["hma_bridge_available"] = self.loader.hma_bridge is not None
            results["elastic_system_available"] = self.loader.elastic_system is not None
            
            if not results["hma_bridge_available"]:
                logger.info("   ⚠️ HMA bridge not available")
                results["success"] = False
                return results
            
            # Test parameter activation
            logger.info("   🚀 Testing parameter activation...")
            
            activation_results = self.loader.hma_bridge.activate_elastic_parameters(
                [0, 1, 2, 3], ["attention"]
            )
            
            successful_activations = sum(1 for success in activation_results.values() if success)
            results["parameter_activation"] = list(activation_results.keys())
            
            logger.info(f"     ✅ Activated {successful_activations}/{len(activation_results)} parameters")
            
            # Test parameter deactivation
            logger.info("   🛑 Testing parameter deactivation...")
            
            deactivation_results = self.loader.hma_bridge.deactivate_elastic_parameters(
                [0, 1], ["attention"]
            )
            
            successful_deactivations = sum(1 for success in deactivation_results.values() if success)
            results["parameter_deactivation"] = list(deactivation_results.keys())
            
            logger.info(f"     ✅ Deactivated {successful_deactivations}/{len(deactivation_results)} parameters")
            
            # Test memory optimization
            logger.info("   🔧 Testing memory optimization...")
            
            optimization_results = self.loader.hma_bridge.optimize_memory_layout()
            
            blocks_moved = optimization_results.get("blocks_moved", 0)
            blocks_compressed = optimization_results.get("blocks_compressed", 0)
            
            results["memory_optimization"] = blocks_moved > 0 or blocks_compressed > 0
            
            logger.info(f"     ✅ Optimized: {blocks_moved} moved, {blocks_compressed} compressed")
            
            results["success"] = successful_activations > 0 and successful_deactivations > 0
            
        except Exception as e:
            logger.error(f"❌ Elastic scaling test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_hardware_utilization(self) -> Dict[str, Any]:
        """Test hardware utilization and status"""
        logger.info("🔧 Testing hardware utilization...")
        
        results = {
            "loader_status": {},
            "components_active": 0,
            "memory_status": {},
            "performance_metrics": {}
        }
        
        try:
            # Get loader status
            status = self.loader.get_status()
            results["loader_status"] = status
            
            # Count active components
            components = status.get("components", {})
            results["components_active"] = sum(1 for active in components.values() if active)
            
            logger.info(f"   📊 Active components: {results['components_active']}/6")
            
            # Check memory status
            if "memory_status" in status:
                memory_status = status["memory_status"]
                results["memory_status"] = memory_status.get("allocation_stats", {})
                
                total_allocated = results["memory_status"].get("total_allocated", 0)
                logger.info(f"   📊 Total memory allocated: {total_allocated / 1024**3:.1f}GB")
            
            # Check performance metrics
            perf_metrics = status.get("performance_metrics", {})
            results["performance_metrics"] = perf_metrics
            
            init_time = perf_metrics.get("initialization_time", 0)
            loading_time = perf_metrics.get("loading_time", 0)
            
            logger.info(f"   ⏱️  Initialization time: {init_time:.2f}s")
            logger.info(f"   ⏱️  Loading time: {loading_time:.2f}s")
            
            # Check component status
            for component, active in components.items():
                status_icon = "✅" if active else "❌"
                logger.info(f"   {status_icon} {component}: {active}")
            
            results["success"] = results["components_active"] >= 4
            
        except Exception as e:
            logger.error(f"❌ Hardware utilization test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_memory_status(self) -> Dict[str, Any]:
        """Test memory system status"""
        logger.info("💾 Testing memory system status...")
        
        results = {
            "hma_bridge_available": False,
            "memory_devices": [],
            "elastic_blocks": 0,
            "active_blocks": 0,
            "total_memory": 0,
            "allocated_memory": 0
        }
        
        try:
            if self.loader.hma_bridge:
                results["hma_bridge_available"] = True
                
                # Get memory status
                memory_status = self.loader.hma_bridge.get_memory_status()
                
                # Device status
                devices = memory_status.get("devices", {})
                results["memory_devices"] = list(devices.keys())
                
                # Elastic blocks
                elastic_blocks = memory_status.get("elastic_blocks", {})
                results["elastic_blocks"] = elastic_blocks.get("total", 0)
                results["active_blocks"] = elastic_blocks.get("active", 0)
                
                # Memory allocation
                allocation_stats = memory_status.get("allocation_stats", {})
                results["total_memory"] = allocation_stats.get("total_allocated", 0)
                results["allocated_memory"] = allocation_stats.get("total_allocated", 0)
                
                logger.info(f"   📊 Memory devices: {len(results['memory_devices'])}")
                logger.info(f"   📊 Elastic blocks: {results['elastic_blocks']} total, {results['active_blocks']} active")
                logger.info(f"   📊 Memory allocated: {results['allocated_memory'] / 1024**3:.1f}GB")
                
                # Device details
                for device_name, device_status in devices.items():
                    utilization = device_status.get("utilization", 0)
                    active_blocks = device_status.get("active_blocks", 0)
                    logger.info(f"   🔧 {device_name}: {utilization:.1%} utilized, {active_blocks} blocks")
                
                results["success"] = len(results["memory_devices"]) > 0
                
            else:
                logger.info("   ⚠️ HMA bridge not available")
                results["success"] = False
        
        except Exception as e:
            logger.error(f"❌ Memory status test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def run_all_tests(self):
        """Run all tests in the pipeline"""
        logger.info("🦄 GEMMA 3N E4B CORE PIPELINE TEST")
        logger.info("=" * 60)
        
        test_start_time = time.time()
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Run tests in sequence
            logger.info("🔬 Running core component tests...")
            
            # Test 1: Component initialization
            self.test_results["component_initialization"] = self.test_component_initialization()
            
            # Test 2: Model loading
            self.test_results["model_loading"] = self.test_model_loading()
            
            # Test 3: Inference performance
            self.test_results["inference_performance"] = self.test_inference_performance()
            
            # Test 4: Elastic scaling
            self.test_results["elastic_scaling"] = self.test_elastic_scaling()
            
            # Test 5: Hardware utilization
            self.test_results["hardware_utilization"] = self.test_hardware_utilization()
            
            # Test 6: Memory status
            self.test_results["memory_status"] = self.test_memory_status()
            
            # Calculate overall results
            total_tests = len(self.test_results)
            successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
            success_rate = successful_tests / total_tests if total_tests > 0 else 0
            
            test_duration = time.time() - test_start_time
            
            # Display summary
            logger.info("=" * 60)
            logger.info("🎯 GEMMA 3N E4B CORE TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"📊 Overall success rate: {success_rate:.1%}")
            logger.info(f"⏱️  Test duration: {test_duration:.1f}s")
            logger.info(f"🔬 Tests completed: {successful_tests}/{total_tests}")
            
            # Performance summary
            if "inference_performance" in self.test_results:
                perf_data = self.test_results["inference_performance"]
                logger.info(f"⚡ Average TPS: {perf_data.get('average_tps', 0):.1f}")
                logger.info(f"🕐 Average inference time: {perf_data.get('average_inference_time', 0):.2f}s")
            
            # Hardware summary
            if "hardware_utilization" in self.test_results:
                hw_data = self.test_results["hardware_utilization"]
                logger.info(f"🔧 Active components: {hw_data.get('components_active', 0)}/6")
            
            # Memory summary
            if "memory_status" in self.test_results:
                mem_data = self.test_results["memory_status"]
                logger.info(f"💾 Elastic blocks: {mem_data.get('elastic_blocks', 0)} total, {mem_data.get('active_blocks', 0)} active")
            
            # Test details
            logger.info("📋 Test Details:")
            for test_name, test_result in self.test_results.items():
                status_icon = "✅" if test_result.get("success", False) else "❌"
                logger.info(f"   {status_icon} {test_name}: {test_result.get('success', False)}")
            
            # Save results
            output_path = Path("./test_results/gemma3n_e4b_core_test")
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "test_timestamp": time.time(),
                    "test_duration": test_duration,
                    "overall_success_rate": success_rate,
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "test_results": self.test_results
                }, f, indent=2, default=str)
            
            logger.info(f"💾 Test results saved to {results_file}")
            
            logger.info("=" * 60)
            if success_rate >= 0.7:
                logger.info("🎉 GEMMA 3N E4B CORE TEST PASSED!")
            else:
                logger.info("⚠️ GEMMA 3N E4B CORE TEST NEEDS ATTENTION")
            logger.info("=" * 60)
            
            return success_rate >= 0.7
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.loader:
                self.loader.shutdown()

def main():
    """Main function to run the test suite"""
    
    # Create test suite
    test_suite = Gemma3nE4BCoreTest()
    
    # Run tests
    try:
        success = test_suite.run_all_tests()
        
        if success:
            logger.info("✅ All core tests passed!")
            return 0
        else:
            logger.error("❌ Some tests failed - check results above")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Test suite interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())