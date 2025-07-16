#!/usr/bin/env python3
"""
REAL NPU Performance Test for Gemma 3 27B
Complete real hardware testing with no simulation
Tests actual tokens per second performance on NPU Phoenix
"""

import torch
import numpy as np
import logging
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Import real NPU kernels
from gemma3_npu_attention_kernel import Gemma3NPUAttentionKernel
from npu_qkv_projection_kernels import NPUQKVProjectionKernels
from npu_scaled_attention_kernel import NPUScaledAttentionKernel

# Import real test data setup
from setup_real_model_test import load_real_test_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealNPUPerformanceTester:
    """Real NPU performance testing with actual hardware execution"""
    
    def __init__(self):
        self.results = {}
        
        # Initialize kernels
        logger.info("🔧 Initializing REAL NPU kernels...")
        self.gemma3_kernel = Gemma3NPUAttentionKernel()
        self.qkv_kernels = NPUQKVProjectionKernels()
        self.scaled_attention = NPUScaledAttentionKernel()
        
        # Load real test data
        logger.info("📂 Loading real quantized model weights...")
        self.weights, self.inputs, self.metadata = load_real_test_data()
        
    def initialize_hardware(self) -> bool:
        """Initialize all NPU kernels with real hardware"""
        logger.info("⚡ Initializing NPU Phoenix hardware...")
        
        start_time = time.time()
        
        if not self.gemma3_kernel.initialize():
            logger.error("❌ Gemma3 kernel initialization failed")
            return False
            
        if not self.qkv_kernels.initialize():
            logger.error("❌ Q/K/V kernels initialization failed")
            return False
            
        if not self.scaled_attention.initialize():
            logger.error("❌ Scaled attention kernel initialization failed")
            return False
        
        init_time = time.time() - start_time
        logger.info(f"✅ All NPU kernels initialized in {init_time:.3f}s")
        
        return True
    
    def test_complete_attention_kernel(self, seq_len: int = 64) -> Dict[str, float]:
        """Test complete Gemma 3 attention kernel end-to-end"""
        logger.info(f"🚀 Testing Complete Attention Kernel (seq_len={seq_len})")
        
        # Get test input
        input_key = f"seq_{seq_len}"
        if input_key not in self.inputs:
            logger.warning(f"⚠️ No test input for seq_len={seq_len}, using seq_64")
            input_key = "seq_64"
        
        hidden_states = torch.from_numpy(self.inputs[input_key])
        
        # Get quantized weights
        q_weight = torch.from_numpy(self.weights['q_weight'])
        q_scale = torch.from_numpy(self.weights['q_scale'])
        k_weight = torch.from_numpy(self.weights['k_weight'])
        k_scale = torch.from_numpy(self.weights['k_scale'])
        v_weight = torch.from_numpy(self.weights['v_weight'])
        v_scale = torch.from_numpy(self.weights['v_scale'])
        o_weight = torch.from_numpy(self.weights['o_weight'])
        o_scale = torch.from_numpy(self.weights['o_scale'])
        
        logger.info(f"📊 Test shapes:")
        logger.info(f"   Hidden states: {hidden_states.shape}")
        logger.info(f"   Q weight: {q_weight.shape}")
        logger.info(f"   K/V weight: {k_weight.shape}/{v_weight.shape}")
        logger.info(f"   O weight: {o_weight.shape}")
        
        # Measure complete attention performance
        num_runs = 5
        times = []
        
        logger.info(f"🔥 Running {num_runs} iterations for accurate measurement...")
        
        for run in range(num_runs):
            start_time = time.time()
            
            try:
                output = self.gemma3_kernel.compute_attention(
                    hidden_states, q_weight, q_scale, k_weight, k_scale,
                    v_weight, v_scale, o_weight, o_scale
                )
                
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
                
                logger.info(f"   Run {run+1}: {run_time*1000:.2f}ms, output: {output.shape}")
                
            except Exception as e:
                logger.error(f"   ❌ Run {run+1} failed: {e}")
                continue
        
        if not times:
            logger.error("❌ All runs failed")
            return {}
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        tokens_per_second = seq_len / avg_time
        
        results = {
            'seq_len': seq_len,
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'tokens_per_second': tokens_per_second,
            'successful_runs': len(times),
            'total_runs': num_runs
        }
        
        logger.info(f"📊 Complete Attention Results (seq_len={seq_len}):")
        logger.info(f"   Average time: {avg_time*1000:.2f}ms")
        logger.info(f"   Min time: {min_time*1000:.2f}ms")
        logger.info(f"   Max time: {max_time*1000:.2f}ms")
        logger.info(f"   🚀 Tokens/second: {tokens_per_second:.2f}")
        logger.info(f"   Success rate: {len(times)}/{num_runs}")
        
        return results
    
    def test_modular_kernels(self, seq_len: int = 64) -> Dict[str, float]:
        """Test modular Q/K/V + attention kernels"""
        logger.info(f"🔧 Testing Modular Kernels (seq_len={seq_len})")
        
        # Get test input
        input_key = f"seq_{seq_len}"
        if input_key not in self.inputs:
            input_key = "seq_64"
        
        hidden_states = torch.from_numpy(self.inputs[input_key])
        
        # Get weights
        q_weight = torch.from_numpy(self.weights['q_weight'])
        q_scale = torch.from_numpy(self.weights['q_scale'])
        k_weight = torch.from_numpy(self.weights['k_weight'])
        k_scale = torch.from_numpy(self.weights['k_scale'])
        v_weight = torch.from_numpy(self.weights['v_weight'])
        v_scale = torch.from_numpy(self.weights['v_scale'])
        o_weight = torch.from_numpy(self.weights['o_weight'])
        
        num_runs = 5
        times = []
        
        logger.info(f"🔥 Running {num_runs} modular kernel iterations...")
        
        for run in range(num_runs):
            start_time = time.time()
            
            try:
                # Step 1: Q/K/V projections
                q, k, v = self.qkv_kernels.execute_qkv_projections(
                    hidden_states, q_weight, q_scale, k_weight, k_scale, v_weight, v_scale
                )
                
                # Step 2: Scaled attention
                context = self.scaled_attention.compute_scaled_attention(q, k, v)
                
                # Step 3: Output projection
                output = torch.matmul(context, o_weight.float().T)
                
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
                
                logger.info(f"   Run {run+1}: {run_time*1000:.2f}ms, output: {output.shape}")
                
            except Exception as e:
                logger.error(f"   ❌ Run {run+1} failed: {e}")
                continue
        
        if not times:
            logger.error("❌ All modular runs failed")
            return {}
        
        # Calculate statistics
        avg_time = np.mean(times)
        tokens_per_second = seq_len / avg_time
        
        results = {
            'seq_len': seq_len,
            'avg_time_ms': avg_time * 1000,
            'tokens_per_second': tokens_per_second,
            'successful_runs': len(times),
            'total_runs': num_runs
        }
        
        logger.info(f"📊 Modular Kernels Results (seq_len={seq_len}):")
        logger.info(f"   Average time: {avg_time*1000:.2f}ms")
        logger.info(f"   🚀 Tokens/second: {tokens_per_second:.2f}")
        
        return results
    
    def test_scaling_performance(self) -> Dict[str, List[Dict]]:
        """Test performance scaling across different sequence lengths"""
        logger.info("📈 Testing Performance Scaling...")
        
        # Test different sequence lengths
        test_lengths = [16, 32, 64, 128, 256]
        
        complete_results = []
        modular_results = []
        
        for seq_len in test_lengths:
            if f"seq_{seq_len}" not in self.inputs:
                logger.warning(f"⚠️ Skipping seq_len={seq_len} (no test data)")
                continue
            
            logger.info(f"🔬 Testing seq_len={seq_len}...")
            
            # Test complete kernel
            complete_result = self.test_complete_attention_kernel(seq_len)
            if complete_result:
                complete_results.append(complete_result)
            
            # Test modular kernels
            modular_result = self.test_modular_kernels(seq_len)
            if modular_result:
                modular_results.append(modular_result)
        
        scaling_results = {
            'complete_kernel': complete_results,
            'modular_kernels': modular_results
        }
        
        logger.info("📊 Scaling Performance Summary:")
        logger.info("Complete Kernel:")
        for result in complete_results:
            logger.info(f"   Seq {result['seq_len']}: {result['tokens_per_second']:.2f} TPS")
        
        logger.info("Modular Kernels:")
        for result in modular_results:
            logger.info(f"   Seq {result['seq_len']}: {result['tokens_per_second']:.2f} TPS")
        
        return scaling_results
    
    def run_complete_test(self) -> Dict:
        """Run complete NPU performance test suite"""
        logger.info("🚀 Starting Complete Real NPU Performance Test")
        logger.info("=" * 60)
        
        # Initialize hardware
        if not self.initialize_hardware():
            logger.error("❌ Hardware initialization failed")
            return {}
        
        # Record test start
        test_start = time.time()
        
        # Test 1: Single sequence performance
        logger.info("\n🔥 Test 1: Single Sequence Performance (64 tokens)")
        single_complete = self.test_complete_attention_kernel(64)
        single_modular = self.test_modular_kernels(64)
        
        # Test 2: Scaling performance
        logger.info("\n🔥 Test 2: Performance Scaling")
        scaling_results = self.test_scaling_performance()
        
        # Compile final results
        total_time = time.time() - test_start
        
        final_results = {
            'test_metadata': {
                'model': self.metadata['model'],
                'test_duration_seconds': total_time,
                'npu_device': 'Phoenix 16 TOPS',
                'test_timestamp': time.time()
            },
            'single_sequence': {
                'complete_kernel': single_complete,
                'modular_kernels': single_modular
            },
            'scaling_performance': scaling_results
        }
        
        # Save results
        results_file = Path("real_npu_performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎉 REAL NPU PERFORMANCE TEST COMPLETE")
        logger.info(f"   Total test time: {total_time:.2f}s")
        logger.info(f"   Results saved to: {results_file}")
        
        # Print best performance
        if single_complete:
            logger.info(f"🚀 BEST PERFORMANCE (Complete): {single_complete['tokens_per_second']:.2f} tokens/sec")
        if single_modular:
            logger.info(f"🚀 BEST PERFORMANCE (Modular): {single_modular['tokens_per_second']:.2f} tokens/sec")
        
        logger.info("=" * 60)
        
        return final_results

def main():
    """Main test execution"""
    logger.info("🦄 Real NPU Performance Tester for Gemma 3 27B")
    logger.info("⚡ Testing REAL hardware performance - NO SIMULATION")
    
    tester = RealNPUPerformanceTester()
    results = tester.run_complete_test()
    
    if results:
        logger.info("✅ Real NPU performance testing completed successfully!")
    else:
        logger.error("❌ Real NPU performance testing failed!")

if __name__ == "__main__":
    main()