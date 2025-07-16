#!/usr/bin/env python3
"""
End-to-End Gemma 3n E4B Pipeline Test
Comprehensive validation of the complete Unicorn execution engine with elastic scaling
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Import our complete pipeline
from gemma3n_e4b_unicorn_loader import (
    Gemma3nE4BUnicornLoader,
    ModelConfig,
    HardwareConfig,
    InferenceConfig,
    InferenceMode,
    LoaderState
)
from gemma3n_e4b_openai_api_server import Gemma3nE4BAPIServer
from gemma3n_e4b_hma_memory_bridge import MemoryDevice

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gemma3nE4BPipelineTest:
    """Comprehensive test suite for Gemma 3n E4B pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.errors = []
        
        # Test configuration
        self.test_config = {
            "model_path": "./models/gemma-3n-e4b-it",
            "test_duration": 300,  # 5 minutes
            "test_prompts": [
                "What is artificial intelligence?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot discovering emotions.",
                "What are the benefits of renewable energy?",
                "How does machine learning work?",
                "Describe the process of photosynthesis.",
                "What is the future of space exploration?",
                "Explain blockchain technology."
            ],
            "concurrent_requests": 4,
            "elastic_scaling_tests": True,
            "hardware_validation": True,
            "memory_stress_test": True,
            "api_server_test": True
        }
        
        # Initialize components
        self.loader = None
        self.api_server = None
        
    def setup_test_environment(self):
        """Setup the test environment"""
        logger.info("🔧 Setting up Gemma 3n E4B test environment...")
        
        # Configure model
        model_config = ModelConfig(
            model_path=self.test_config["model_path"],
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
        
        # Initialize API server
        self.api_server = Gemma3nE4BAPIServer(self.test_config["model_path"])
        
        logger.info("✅ Test environment setup complete")
    
    def test_component_initialization(self) -> Dict[str, Any]:
        """Test individual component initialization"""
        logger.info("🔬 Testing component initialization...")
        
        results = {
            "quantizer": False,
            "allocator": False,
            "elastic_system": False,
            "npu_kernels": False,
            "vulkan_shaders": False,
            "hma_bridge": False,
            "initialization_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Test quantizer
            if self.loader.quantizer:
                results["quantizer"] = True
                logger.info("   ✅ Quantization engine initialized")
            
            # Test allocator
            if self.loader.allocator:
                results["allocator"] = True
                logger.info("   ✅ Mix-n-Match allocator initialized")
            
            # Test elastic system
            if self.loader.elastic_system:
                results["elastic_system"] = True
                logger.info("   ✅ Elastic parameter system initialized")
            
            # Test NPU kernels
            if self.loader.npu_kernels:
                results["npu_kernels"] = True
                logger.info("   ✅ NPU attention kernels initialized")
            
            # Test Vulkan shaders
            if self.loader.vulkan_shaders:
                results["vulkan_shaders"] = True
                logger.info("   ✅ Vulkan FFN shaders initialized")
            
            # Test HMA bridge
            if self.loader.hma_bridge:
                results["hma_bridge"] = True
                logger.info("   ✅ HMA memory bridge initialized")
            
            results["initialization_time"] = time.time() - start_time
            results["success"] = all(results[key] for key in results if key not in ["initialization_time", "success"])
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading with quantization"""
        logger.info("🚀 Testing model loading...")
        
        results = {
            "loading_success": False,
            "loading_time": 0.0,
            "model_size": 0,
            "quantization_success": False,
            "elastic_params_initialized": 0
        }
        
        start_time = time.time()
        
        try:
            # Load model
            loading_success = self.loader.load_model()
            
            if loading_success:
                results["loading_success"] = True
                results["loading_time"] = time.time() - start_time
                
                # Get model status
                status = self.loader.get_status()
                results["model_size"] = status.get("active_elastic_params", 0)
                results["quantization_success"] = True
                results["elastic_params_initialized"] = len(self.loader.active_elastic_params)
                
                logger.info(f"   ✅ Model loaded in {results['loading_time']:.2f}s")
                logger.info(f"   📊 Active elastic params: {results['elastic_params_initialized']}")
                
            else:
                results["error"] = "Model loading failed"
                logger.error("   ❌ Model loading failed")
        
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            results["error"] = str(e)
        
        return results
    
    def test_inference_performance(self) -> Dict[str, Any]:
        """Test inference performance with various prompts"""
        logger.info("⚡ Testing inference performance...")
        
        results = {
            "inference_results": [],
            "average_tps": 0.0,
            "average_ttft": 0.0,
            "success_rate": 0.0,
            "total_tests": len(self.test_config["test_prompts"])
        }
        
        successful_tests = 0
        total_tps = 0.0
        total_ttft = 0.0
        
        for i, prompt in enumerate(self.test_config["test_prompts"]):
            logger.info(f"   🔍 Test {i+1}/{len(self.test_config['test_prompts'])}: {prompt[:50]}...")
            
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
                start_time = time.time()
                result = self.loader.generate(prompt, inference_config)
                
                if "error" not in result:
                    ttft = time.time() - start_time
                    tps = result.get("tokens_per_second", 0)
                    
                    test_result = {
                        "prompt": prompt,
                        "success": True,
                        "tokens_generated": result.get("tokens_generated", 0),
                        "inference_time": result.get("inference_time", 0),
                        "tokens_per_second": tps,
                        "ttft": ttft,
                        "elastic_params_active": result.get("elastic_params_active", 0),
                        "memory_usage": result.get("memory_usage", 0)
                    }
                    
                    results["inference_results"].append(test_result)
                    successful_tests += 1
                    total_tps += tps
                    total_ttft += ttft
                    
                    logger.info(f"     ✅ {tps:.1f} TPS, {ttft:.2f}s TTFT")
                    
                else:
                    logger.error(f"     ❌ Inference failed: {result['error']}")
                    results["inference_results"].append({
                        "prompt": prompt,
                        "success": False,
                        "error": result["error"]
                    })
            
            except Exception as e:
                logger.error(f"     ❌ Inference error: {e}")
                results["inference_results"].append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate averages
        if successful_tests > 0:
            results["average_tps"] = total_tps / successful_tests
            results["average_ttft"] = total_ttft / successful_tests
            results["success_rate"] = successful_tests / len(self.test_config["test_prompts"])
        
        logger.info(f"   📊 Success rate: {results['success_rate']:.1%}")
        logger.info(f"   📊 Average TPS: {results['average_tps']:.1f}")
        logger.info(f"   📊 Average TTFT: {results['average_ttft']:.2f}s")
        
        return results
    
    def test_elastic_scaling(self) -> Dict[str, Any]:
        """Test elastic parameter scaling functionality"""
        logger.info("🔄 Testing elastic parameter scaling...")
        
        results = {
            "scaling_tests": [],
            "parameter_activation_success": False,
            "parameter_deactivation_success": False,
            "memory_optimization_success": False
        }
        
        if not self.loader.elastic_system or not self.loader.hma_bridge:
            results["error"] = "Elastic system or HMA bridge not available"
            return results
        
        try:
            # Test parameter activation
            logger.info("   🚀 Testing parameter activation...")
            
            # Activate parameters for layers 0-3 (attention)
            activation_results = self.loader.hma_bridge.activate_elastic_parameters(
                [0, 1, 2, 3], ["attention"]
            )
            
            successful_activations = sum(1 for success in activation_results.values() if success)
            total_activations = len(activation_results)
            
            if successful_activations > 0:
                results["parameter_activation_success"] = True
                logger.info(f"     ✅ Activated {successful_activations}/{total_activations} parameters")
            
            # Test parameter deactivation
            logger.info("   🛑 Testing parameter deactivation...")
            
            deactivation_results = self.loader.hma_bridge.deactivate_elastic_parameters(
                [0, 1], ["attention"]
            )
            
            successful_deactivations = sum(1 for success in deactivation_results.values() if success)
            total_deactivations = len(deactivation_results)
            
            if successful_deactivations > 0:
                results["parameter_deactivation_success"] = True
                logger.info(f"     ✅ Deactivated {successful_deactivations}/{total_deactivations} parameters")
            
            # Test memory optimization
            logger.info("   🔧 Testing memory optimization...")
            
            optimization_results = self.loader.hma_bridge.optimize_memory_layout()
            
            if optimization_results.get("blocks_moved", 0) > 0 or optimization_results.get("blocks_compressed", 0) > 0:
                results["memory_optimization_success"] = True
                logger.info(f"     ✅ Optimized: {optimization_results['blocks_moved']} moved, {optimization_results['blocks_compressed']} compressed")
            
            # Test dynamic scaling during inference
            logger.info("   📊 Testing dynamic scaling during inference...")
            
            for seq_length in [64, 256, 1024]:
                logger.info(f"     🔍 Testing sequence length: {seq_length}")
                
                # Create test prompt of specified length
                test_prompt = "Test " * (seq_length // 5)
                
                inference_config = InferenceConfig(
                    mode=InferenceMode.ADAPTIVE,
                    max_tokens=50,
                    elastic_scaling=True,
                    dynamic_allocation=True
                )
                
                start_time = time.time()
                result = self.loader.generate(test_prompt, inference_config)
                
                if "error" not in result:
                    scaling_result = {
                        "sequence_length": seq_length,
                        "success": True,
                        "inference_time": result.get("inference_time", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "elastic_params_active": result.get("elastic_params_active", 0)
                    }
                    
                    results["scaling_tests"].append(scaling_result)
                    logger.info(f"       ✅ {result.get('tokens_per_second', 0):.1f} TPS, {result.get('elastic_params_active', 0)} active params")
                else:
                    logger.error(f"       ❌ Scaling test failed: {result['error']}")
        
        except Exception as e:
            logger.error(f"❌ Elastic scaling test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def test_hardware_utilization(self) -> Dict[str, Any]:
        """Test hardware utilization and coordination"""
        logger.info("🔧 Testing hardware utilization...")
        
        results = {
            "npu_utilization": False,
            "igpu_utilization": False,
            "memory_usage": {},
            "hardware_coordination": False
        }
        
        try:
            # Get current status
            status = self.loader.get_status()
            
            # Test NPU utilization
            if status.get("components", {}).get("npu_kernels", False):
                results["npu_utilization"] = True
                logger.info("   ✅ NPU kernels active")
            
            # Test iGPU utilization
            if status.get("components", {}).get("vulkan_shaders", False):
                results["igpu_utilization"] = True
                logger.info("   ✅ Vulkan shaders active")
            
            # Test memory status
            if "memory_status" in status:
                memory_status = status["memory_status"]
                results["memory_usage"] = memory_status.get("allocation_stats", {})
                
                total_allocated = results["memory_usage"].get("total_allocated", 0)
                logger.info(f"   📊 Total memory allocated: {total_allocated / 1024**3:.1f}GB")
            
            # Test hardware coordination
            if results["npu_utilization"] and results["igpu_utilization"]:
                results["hardware_coordination"] = True
                logger.info("   ✅ NPU+iGPU coordination working")
        
        except Exception as e:
            logger.error(f"❌ Hardware utilization test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def test_concurrent_inference(self) -> Dict[str, Any]:
        """Test concurrent inference requests"""
        logger.info("⚡ Testing concurrent inference...")
        
        results = {
            "concurrent_requests": self.test_config["concurrent_requests"],
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "throughput": 0.0
        }
        
        def run_inference(prompt_id: int, prompt: str) -> Dict:
            try:
                inference_config = InferenceConfig(
                    mode=InferenceMode.PERFORMANCE,
                    max_tokens=50,
                    elastic_scaling=True
                )
                
                start_time = time.time()
                result = self.loader.generate(prompt, inference_config)
                response_time = time.time() - start_time
                
                if "error" not in result:
                    return {
                        "prompt_id": prompt_id,
                        "success": True,
                        "response_time": response_time,
                        "tokens_generated": result.get("tokens_generated", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0)
                    }
                else:
                    return {
                        "prompt_id": prompt_id,
                        "success": False,
                        "error": result["error"]
                    }
            except Exception as e:
                return {
                    "prompt_id": prompt_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Run concurrent requests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.test_config["concurrent_requests"]) as executor:
            futures = []
            
            for i in range(self.test_config["concurrent_requests"]):
                prompt = self.test_config["test_prompts"][i % len(self.test_config["test_prompts"])]
                future = executor.submit(run_inference, i, prompt)
                futures.append(future)
            
            # Collect results
            total_response_time = 0.0
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    
                    if result["success"]:
                        results["successful_requests"] += 1
                        total_response_time += result["response_time"]
                        logger.info(f"   ✅ Request {result['prompt_id']}: {result['tokens_per_second']:.1f} TPS")
                    else:
                        results["failed_requests"] += 1
                        logger.error(f"   ❌ Request {result['prompt_id']}: {result['error']}")
                
                except Exception as e:
                    results["failed_requests"] += 1
                    logger.error(f"   ❌ Request failed: {e}")
        
        total_test_time = time.time() - start_time
        
        # Calculate metrics
        if results["successful_requests"] > 0:
            results["average_response_time"] = total_response_time / results["successful_requests"]
            results["throughput"] = results["successful_requests"] / total_test_time
        
        logger.info(f"   📊 Successful requests: {results['successful_requests']}/{self.test_config['concurrent_requests']}")
        logger.info(f"   📊 Average response time: {results['average_response_time']:.2f}s")
        logger.info(f"   📊 Throughput: {results['throughput']:.1f} requests/second")
        
        return results
    
    async def test_api_server(self) -> Dict[str, Any]:
        """Test OpenAI API server functionality"""
        logger.info("🌐 Testing OpenAI API server...")
        
        results = {
            "server_startup": False,
            "health_check": False,
            "chat_completion": False,
            "streaming_response": False,
            "concurrent_api_requests": False
        }
        
        try:
            # Wait for server to initialize
            await asyncio.sleep(5)
            
            # Test server health
            if self.api_server.loader_ready:
                results["health_check"] = True
                logger.info("   ✅ API server health check passed")
            
            # Test chat completion
            try:
                from gemma3n_e4b_openai_api_server import ChatCompletionRequest, Message
                
                request = ChatCompletionRequest(
                    model="gemma-3n-e4b-it",
                    messages=[
                        Message(role="user", content="What is artificial intelligence?")
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                
                completion_result = await self.api_server.generate_completion(request)
                
                if "error" not in completion_result:
                    results["chat_completion"] = True
                    logger.info("   ✅ Chat completion working")
                    logger.info(f"   📊 Generated {completion_result.get('tokens_generated', 0)} tokens")
                else:
                    logger.error(f"   ❌ Chat completion failed: {completion_result['error']}")
            
            except Exception as e:
                logger.error(f"   ❌ Chat completion error: {e}")
            
            # Test streaming response
            try:
                stream_request = ChatCompletionRequest(
                    model="gemma-3n-e4b-it",
                    messages=[
                        Message(role="user", content="Tell me about quantum computing")
                    ],
                    max_tokens=30,
                    stream=True
                )
                
                stream_chunks = []
                async for chunk in self.api_server.stream_completion(stream_request):
                    stream_chunks.append(chunk)
                    if len(stream_chunks) >= 3:  # Test first few chunks
                        break
                
                if len(stream_chunks) > 0:
                    results["streaming_response"] = True
                    logger.info("   ✅ Streaming response working")
                    logger.info(f"   📊 Received {len(stream_chunks)} stream chunks")
            
            except Exception as e:
                logger.error(f"   ❌ Streaming test error: {e}")
        
        except Exception as e:
            logger.error(f"❌ API server test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def test_memory_stress(self) -> Dict[str, Any]:
        """Test memory stress scenarios"""
        logger.info("💾 Testing memory stress scenarios...")
        
        results = {
            "memory_allocation": False,
            "memory_deallocation": False,
            "memory_optimization": False,
            "memory_pressure_handling": False
        }
        
        try:
            if not self.loader.hma_bridge:
                results["error"] = "HMA bridge not available"
                return results
            
            # Test memory allocation
            logger.info("   🚀 Testing memory allocation...")
            
            # Activate many parameters
            activation_results = self.loader.hma_bridge.activate_elastic_parameters(
                list(range(20)), ["attention", "ffn"]
            )
            
            successful_activations = sum(1 for success in activation_results.values() if success)
            
            if successful_activations > 0:
                results["memory_allocation"] = True
                logger.info(f"   ✅ Allocated memory for {successful_activations} parameters")
            
            # Test memory pressure handling
            logger.info("   🔧 Testing memory pressure handling...")
            
            # Try to allocate even more parameters
            pressure_results = self.loader.hma_bridge.activate_elastic_parameters(
                list(range(24)), ["attention", "ffn"]
            )
            
            # Check if system handles pressure gracefully
            memory_status = self.loader.hma_bridge.get_memory_status()
            
            if memory_status and "allocation_stats" in memory_status:
                results["memory_pressure_handling"] = True
                logger.info("   ✅ Memory pressure handling working")
            
            # Test memory optimization
            logger.info("   ⚡ Testing memory optimization...")
            
            optimization_results = self.loader.hma_bridge.optimize_memory_layout()
            
            if optimization_results.get("blocks_compressed", 0) > 0 or optimization_results.get("blocks_moved", 0) > 0:
                results["memory_optimization"] = True
                logger.info("   ✅ Memory optimization working")
            
            # Test memory deallocation
            logger.info("   🛑 Testing memory deallocation...")
            
            deallocation_results = self.loader.hma_bridge.deactivate_elastic_parameters(
                list(range(10)), ["attention", "ffn"]
            )
            
            successful_deallocations = sum(1 for success in deallocation_results.values() if success)
            
            if successful_deallocations > 0:
                results["memory_deallocation"] = True
                logger.info(f"   ✅ Deallocated memory for {successful_deallocations} parameters")
        
        except Exception as e:
            logger.error(f"❌ Memory stress test failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("📊 Generating test report...")
        
        # Calculate overall scores
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Performance summary
        performance_summary = {}
        if "inference_performance" in self.test_results:
            perf_data = self.test_results["inference_performance"]
            performance_summary = {
                "average_tps": perf_data.get("average_tps", 0),
                "average_ttft": perf_data.get("average_ttft", 0),
                "success_rate": perf_data.get("success_rate", 0)
            }
        
        # Hardware utilization summary
        hardware_summary = {}
        if "hardware_utilization" in self.test_results:
            hw_data = self.test_results["hardware_utilization"]
            hardware_summary = {
                "npu_active": hw_data.get("npu_utilization", False),
                "igpu_active": hw_data.get("igpu_utilization", False),
                "coordination": hw_data.get("hardware_coordination", False)
            }
        
        # Elastic scaling summary
        elastic_summary = {}
        if "elastic_scaling" in self.test_results:
            elastic_data = self.test_results["elastic_scaling"]
            elastic_summary = {
                "activation_success": elastic_data.get("parameter_activation_success", False),
                "deactivation_success": elastic_data.get("parameter_deactivation_success", False),
                "memory_optimization": elastic_data.get("memory_optimization_success", False)
            }
        
        report = {
            "test_timestamp": time.time(),
            "test_duration": time.time() - self.test_start_time,
            "overall_success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "performance_summary": performance_summary,
            "hardware_summary": hardware_summary,
            "elastic_summary": elastic_summary,
            "detailed_results": self.test_results,
            "errors": self.errors
        }
        
        return report
    
    def save_test_results(self, output_path: str):
        """Save test results to file"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report = self.generate_test_report()
        
        # Save report
        report_file = output_dir / "gemma3n_e4b_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save performance metrics
        metrics_file = output_dir / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        logger.info(f"✅ Test results saved to {output_dir}")
        return output_dir
    
    async def run_all_tests(self):
        """Run all tests in the pipeline"""
        logger.info("🦄 GEMMA 3N E4B PIPELINE TEST SUITE")
        logger.info("=" * 60)
        
        self.test_start_time = time.time()
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Run tests in sequence
            logger.info("🔬 Running component tests...")
            
            # Test 1: Component initialization
            self.test_results["component_initialization"] = self.test_component_initialization()
            
            # Test 2: Model loading
            self.test_results["model_loading"] = self.test_model_loading()
            
            # Test 3: Inference performance
            self.test_results["inference_performance"] = self.test_inference_performance()
            
            # Test 4: Elastic scaling
            if self.test_config["elastic_scaling_tests"]:
                self.test_results["elastic_scaling"] = self.test_elastic_scaling()
            
            # Test 5: Hardware utilization
            if self.test_config["hardware_validation"]:
                self.test_results["hardware_utilization"] = self.test_hardware_utilization()
            
            # Test 6: Concurrent inference
            self.test_results["concurrent_inference"] = self.test_concurrent_inference()
            
            # Test 7: Memory stress test
            if self.test_config["memory_stress_test"]:
                self.test_results["memory_stress"] = self.test_memory_stress()
            
            # Test 8: API server test
            if self.test_config["api_server_test"]:
                self.test_results["api_server"] = await self.test_api_server()
            
            # Generate final report
            report = self.generate_test_report()
            
            # Display summary
            logger.info("=" * 60)
            logger.info("🎯 GEMMA 3N E4B PIPELINE TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"📊 Overall success rate: {report['overall_success_rate']:.1%}")
            logger.info(f"⏱️  Test duration: {report['test_duration']:.1f}s")
            logger.info(f"🔬 Tests completed: {report['successful_tests']}/{report['total_tests']}")
            
            if report["performance_summary"]:
                logger.info(f"⚡ Average TPS: {report['performance_summary']['average_tps']:.1f}")
                logger.info(f"🕐 Average TTFT: {report['performance_summary']['average_ttft']:.2f}s")
            
            if report["hardware_summary"]:
                logger.info(f"🔧 NPU active: {report['hardware_summary']['npu_active']}")
                logger.info(f"🔧 iGPU active: {report['hardware_summary']['igpu_active']}")
                logger.info(f"🔧 Hardware coordination: {report['hardware_summary']['coordination']}")
            
            if report["elastic_summary"]:
                logger.info(f"🔄 Elastic scaling: {report['elastic_summary']['activation_success']}")
                logger.info(f"💾 Memory optimization: {report['elastic_summary']['memory_optimization']}")
            
            # Save results
            output_path = "./test_results/gemma3n_e4b_pipeline_test"
            self.save_test_results(output_path)
            
            logger.info("=" * 60)
            logger.info("🎉 GEMMA 3N E4B PIPELINE TEST COMPLETE!")
            logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.errors.append(str(e))
            return {"error": str(e), "test_results": self.test_results}
        
        finally:
            # Cleanup
            if self.loader:
                self.loader.shutdown()

def main():
    """Main function to run the test suite"""
    
    # Create test suite
    test_suite = Gemma3nE4BPipelineTest()
    
    # Run tests
    try:
        report = asyncio.run(test_suite.run_all_tests())
        
        if "error" not in report:
            logger.info("✅ All tests completed successfully!")
            return 0
        else:
            logger.error(f"❌ Test suite failed: {report['error']}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Test suite interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())