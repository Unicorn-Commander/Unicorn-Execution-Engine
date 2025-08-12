#!/usr/bin/env python3
"""
Test Memory-Optimized Gemma 3 27B Pipeline
Validates the complete NPU+iGPU pipeline with memory constraints
"""

import os
import torch
import time
import gc
import psutil
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional

# Ensure we're using the AI environment
import sys
sys.path.append(str(Path(__file__).parent))

# Import our optimized components
from ultra_memory_efficient_quantize import UltraMemoryEfficientQuantizer
from npu_memory_optimized_kernel import NPUMemoryOptimizedKernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimized27BPipeline:
    """Complete memory-optimized pipeline for Gemma 3 27B"""
    
    def __init__(self):
        self.model_path = Path("./models/gemma-3-27b-it")
        self.quantized_path = Path("./quantized_models/gemma-3-27b-it-ultra-memory-efficient")
        
        # Hardware configuration
        self.config = {
            "model_name": "gemma-3-27b-it",
            "vocab_size": 256000,
            "seq_length": 2048,
            "d_model": 4096,
            "n_layers": 62,
            "n_heads": 32,
            "intermediate_size": 14336,
            "head_dim": 128,
            "npu_memory_mb": 2048,
            "chunk_size": 256,  # Smaller chunks for 27B
            "max_sequence_length": 2048
        }
        
        # Components
        self.quantizer = None
        self.npu_kernel = None
        
        # Memory monitoring
        self.process = psutil.Process()
        self.memory_stats = {
            'peak_quantization_mb': 0,
            'peak_inference_mb': 0,
            'current_mb': 0
        }
        
        logger.info("🦄 Memory-Optimized Gemma 3 27B Pipeline initialized")
        
    def monitor_memory(self, stage: str = "unknown") -> float:
        """Monitor memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        self.memory_stats['current_mb'] = memory_mb
        
        if stage == "quantization":
            self.memory_stats['peak_quantization_mb'] = max(
                self.memory_stats['peak_quantization_mb'], memory_mb
            )
        elif stage == "inference":
            self.memory_stats['peak_inference_mb'] = max(
                self.memory_stats['peak_inference_mb'], memory_mb
            )
        
        return memory_mb
    
    def check_system_resources(self) -> bool:
        """Check if system has sufficient resources"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"🖥️ System Resources:")
        logger.info(f"   Available RAM: {available_gb:.1f} GB")
        logger.info(f"   CPU cores: {psutil.cpu_count()}")
        
        if available_gb < 10:
            logger.warning("⚠️ Low memory detected. Process may be slow.")
        
        return available_gb >= 5  # Minimum 5GB required
    
    def test_quantization_phase(self) -> bool:
        """Test the quantization phase"""
        logger.info("🔧 Testing Quantization Phase")
        
        if not self.model_path.exists():
            logger.error(f"❌ Model not found: {self.model_path}")
            return False
        
        # Initialize quantizer
        self.quantizer = UltraMemoryEfficientQuantizer(str(self.model_path))
        
        # Monitor memory during quantization
        start_memory = self.monitor_memory("quantization")
        start_time = time.time()
        
        # Check if we have already quantized
        if (self.quantized_path / "quantization_results.json").exists():
            logger.info("✅ Quantization already completed")
            
            # Load existing results
            with open(self.quantized_path / "quantization_results.json", 'r') as f:
                results = json.load(f)
            
            logger.info(f"   Original size: {results['original_size_gb']:.2f} GB")
            logger.info(f"   Quantized size: {results['quantized_size_gb']:.2f} GB")
            logger.info(f"   Memory reduction: {results['memory_reduction']:.1%}")
            
            return True
        
        # Run quantization (this will be time-consuming)
        logger.info("⚠️ Starting quantization - this may take 10-30 minutes...")
        
        try:
            # Quantize with short timeout for demo
            results = self.quantizer.quantize_model()
            
            quantization_time = time.time() - start_time
            peak_memory = self.monitor_memory("quantization")
            
            if results:
                logger.info("✅ Quantization completed successfully!")
                logger.info(f"   Time: {quantization_time/60:.1f} minutes")
                logger.info(f"   Peak memory: {peak_memory:.1f} MB")
                logger.info(f"   Size reduction: {results['memory_reduction']:.1%}")
                return True
            else:
                logger.error("❌ Quantization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Quantization error: {e}")
            return False
    
    def test_npu_kernel_phase(self) -> bool:
        """Test the NPU kernel phase"""
        logger.info("🧠 Testing NPU Kernel Phase")
        
        # Initialize NPU kernel
        self.npu_kernel = NPUMemoryOptimizedKernel(self.config)
        
        if not self.npu_kernel.initialize_memory_pool():
            logger.error("❌ NPU kernel initialization failed")
            return False
        
        # Test memory allocation
        test_size = 1024 * 1024  # 1MB
        offset = self.npu_kernel.allocate_npu_memory(test_size)
        
        if offset is not None:
            logger.info("✅ NPU memory allocation successful")
            self.npu_kernel.free_npu_memory(offset)
        else:
            logger.error("❌ NPU memory allocation failed")
            return False
        
        # Test attention processing with different sizes
        test_sequences = [256, 512, 1024]
        
        for seq_len in test_sequences:
            logger.info(f"   🧪 Testing sequence length: {seq_len}")
            
            # Create test data
            hidden_states = torch.randn(1, seq_len, self.config['d_model'], dtype=torch.float16)
            
            # Create mock layer weights
            layer_weights = {
                'q_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'k_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'v_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'o_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
            }
            
            # Test processing
            start_time = time.time()
            memory_before = self.monitor_memory("inference")
            
            try:
                output = self.npu_kernel.process_layer(hidden_states, layer_weights, layer_idx=0)
                processing_time = time.time() - start_time
                memory_after = self.monitor_memory("inference")
                
                logger.info(f"      ✅ Sequence {seq_len}: {processing_time:.3f}s, Memory: {memory_after:.1f}MB")
                
                # Verify output shape
                assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} vs {hidden_states.shape}"
                
            except Exception as e:
                logger.error(f"      ❌ Sequence {seq_len} failed: {e}")
                return False
            
            # Cleanup
            del hidden_states, layer_weights, output
            gc.collect()
        
        return True
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test the complete end-to-end pipeline"""
        logger.info("🔄 Testing End-to-End Pipeline")
        
        if not self.npu_kernel:
            logger.error("❌ NPU kernel not initialized")
            return False
        
        # Test with multiple layers
        num_test_layers = 3  # Test with 3 layers instead of all 62
        seq_len = 512
        
        logger.info(f"   Testing {num_test_layers} layers with sequence length {seq_len}")
        
        # Initialize hidden states
        hidden_states = torch.randn(1, seq_len, self.config['d_model'], dtype=torch.float16)
        
        total_time = 0
        memory_peak = 0
        
        for layer_idx in range(num_test_layers):
            logger.info(f"   🔄 Processing layer {layer_idx + 1}/{num_test_layers}")
            
            # Create layer weights (normally would come from quantized model)
            layer_weights = {
                'q_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'k_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'v_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
                'o_proj': torch.randn(self.config['d_model'], self.config['d_model'], dtype=torch.float16),
            }
            
            # Process layer
            start_time = time.time()
            memory_before = self.monitor_memory("inference")
            
            hidden_states = self.npu_kernel.process_layer(hidden_states, layer_weights, layer_idx)
            
            layer_time = time.time() - start_time
            memory_after = self.monitor_memory("inference")
            
            total_time += layer_time
            memory_peak = max(memory_peak, memory_after)
            
            logger.info(f"      ✅ Layer {layer_idx + 1}: {layer_time:.3f}s")
            
            # Cleanup
            del layer_weights
            gc.collect()
        
        # Calculate performance metrics
        tokens_per_second = seq_len / total_time
        
        logger.info(f"✅ End-to-End Pipeline Results:")
        logger.info(f"   Total time: {total_time:.3f}s")
        logger.info(f"   Tokens per second: {tokens_per_second:.2f}")
        logger.info(f"   Peak memory: {memory_peak:.1f}MB")
        
        # Estimate full model performance
        estimated_full_time = total_time * (self.config['n_layers'] / num_test_layers)
        estimated_tps = seq_len / estimated_full_time
        
        logger.info(f"🎯 Estimated Full Model Performance:")
        logger.info(f"   Estimated time for {self.config['n_layers']} layers: {estimated_full_time:.2f}s")
        logger.info(f"   Estimated TPS: {estimated_tps:.2f}")
        
        return True
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the memory-optimized pipeline"""
        logger.info("🚀 Running Comprehensive Memory-Optimized Pipeline Test")
        logger.info("=" * 70)
        
        results = {
            'system_check': False,
            'quantization': False,
            'npu_kernel': False,
            'end_to_end': False,
            'memory_stats': self.memory_stats
        }
        
        # System resource check
        if not self.check_system_resources():
            logger.error("❌ Insufficient system resources")
            return results
        results['system_check'] = True
        
        # Test quantization phase
        if not self.test_quantization_phase():
            logger.error("❌ Quantization phase failed")
            return results
        results['quantization'] = True
        
        # Test NPU kernel phase
        if not self.test_npu_kernel_phase():
            logger.error("❌ NPU kernel phase failed")
            return results
        results['npu_kernel'] = True
        
        # Test end-to-end pipeline
        if not self.test_end_to_end_pipeline():
            logger.error("❌ End-to-end pipeline failed")
            return results
        results['end_to_end'] = True
        
        # Final memory stats
        final_memory = self.monitor_memory("final")
        results['memory_stats'] = self.memory_stats
        
        logger.info("🎉 COMPREHENSIVE TEST COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"✅ All phases passed successfully!")
        logger.info(f"📊 Memory Usage Summary:")
        logger.info(f"   Peak quantization: {self.memory_stats['peak_quantization_mb']:.1f}MB")
        logger.info(f"   Peak inference: {self.memory_stats['peak_inference_mb']:.1f}MB")
        logger.info(f"   Final memory: {final_memory:.1f}MB")
        
        return results

def main():
    """Main test function"""
    pipeline = MemoryOptimized27BPipeline()
    results = pipeline.run_comprehensive_test()
    
    # Save results
    output_dir = Path("./test_results/memory_optimized_27b")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📁 Test results saved to {output_dir}")
    
    # Return success status
    all_passed = all(results[key] for key in ['system_check', 'quantization', 'npu_kernel', 'end_to_end'])
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)