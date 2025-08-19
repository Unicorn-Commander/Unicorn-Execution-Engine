#!/usr/bin/env python3
"""
CPU Baseline Performance Measurement for UC1-NPU-Production
Establishes exact performance targets for NPU optimization
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_loader import Qwen3EmbeddingLoader
import numpy as np
import time
import json

class CPUBaselineBenchmark:
    """Comprehensive CPU performance benchmarking"""
    
    def __init__(self):
        self.loader = Qwen3EmbeddingLoader()
        self.results = {}
        
    def setup(self):
        """Setup model and verify functionality"""
        
        print("🔧 SETTING UP CPU BASELINE BENCHMARK")
        print("=" * 50)
        
        # Load model
        if not self.loader.load_model():
            raise RuntimeError("Failed to load Qwen3 model")
        
        # Get model info
        self.model_info = self.loader.get_model_info()
        print(f"✅ Model loaded: {self.model_info['parameters']:,} parameters")
        print(f"✅ Memory usage: {self.model_info['memory_gb']:.2f} GB")
        
    def benchmark_single_embeddings(self):
        """Benchmark single embedding generation"""
        
        print(f"\n📊 SINGLE EMBEDDING BENCHMARK")
        print("-" * 30)
        
        test_texts = [
            "Short text",
            "This is a medium length sentence for testing embedding generation speed",
            "This is a much longer text that contains multiple sentences and should test the model's ability to handle longer input sequences. It includes various types of content to ensure comprehensive testing of the embedding generation process.",
            "Machine learning and artificial intelligence",
            "Natural language processing with transformer models",
            "Deep neural networks for computer vision applications",
            "Distributed computing and cloud infrastructure",
            "Quantum computing and quantum algorithms research",
            "Bioinformatics and computational biology methods",
            "Financial modeling and quantitative analysis"
        ]
        
        times = []
        embedding_shapes = []
        
        # Warm-up
        _ = self.loader.generate_embedding_cpu(test_texts[0])
        
        for i, text in enumerate(test_texts):
            start_time = time.perf_counter()
            embedding = self.loader.generate_embedding_cpu(text)
            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            embedding_shapes.append(embedding.shape)
            
            print(f"Text {i+1:2}: {elapsed_ms:6.2f}ms - {text[:40]}...")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        self.results['single_embeddings'] = {
            'avg_latency_ms': avg_time,
            'std_latency_ms': std_time,
            'min_latency_ms': min_time,
            'max_latency_ms': max_time,
            'throughput_per_sec': 1000 / avg_time,
            'all_times_ms': times,
            'embedding_shapes': [str(shape) for shape in embedding_shapes]
        }
        
        print(f"\n🎯 SINGLE EMBEDDING RESULTS:")
        print(f"   Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"   Range: {min_time:.2f}ms - {max_time:.2f}ms") 
        print(f"   Throughput: {1000/avg_time:.1f} embeddings/sec")
        
    def benchmark_batch_embeddings(self):
        """Benchmark batch embedding generation"""
        
        print(f"\n📊 BATCH EMBEDDING BENCHMARK")
        print("-" * 30)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        
        base_texts = [
            "Machine learning algorithms for data science",
            "Natural language processing and text analysis", 
            "Computer vision and image recognition systems",
            "Artificial intelligence in healthcare applications",
            "Deep learning for autonomous vehicle systems",
            "Quantum computing and quantum machine learning",
            "Distributed computing and parallel processing",
            "Cybersecurity and network protection methods"
        ]
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}:")
            
            # Create batch by repeating texts
            batch_texts = (base_texts * ((batch_size // len(base_texts)) + 1))[:batch_size]
            
            times = []
            
            # Run multiple tests for this batch size
            for run in range(5):
                start_time = time.perf_counter()
                
                embeddings = []
                for text in batch_texts:
                    embedding = self.loader.generate_embedding_cpu(text)
                    embeddings.append(embedding)
                
                end_time = time.perf_counter()
                
                total_time_ms = (end_time - start_time) * 1000
                per_embedding_ms = total_time_ms / batch_size
                times.append(per_embedding_ms)
                
                if run == 0:
                    print(f"   Total time: {total_time_ms:.2f}ms")
                    print(f"   Per embedding: {per_embedding_ms:.2f}ms")
            
            avg_per_embedding = np.mean(times)
            std_per_embedding = np.std(times)
            
            batch_results[batch_size] = {
                'avg_per_embedding_ms': avg_per_embedding,
                'std_per_embedding_ms': std_per_embedding,
                'throughput_per_sec': 1000 / avg_per_embedding,
                'all_times_ms': times
            }
            
            print(f"   Average per embedding: {avg_per_embedding:.2f}ms ± {std_per_embedding:.2f}ms")
            print(f"   Throughput: {1000/avg_per_embedding:.1f} embeddings/sec")
        
        self.results['batch_embeddings'] = batch_results
        
        # Find optimal batch size
        best_batch = min(batch_results.keys(), 
                        key=lambda x: batch_results[x]['avg_per_embedding_ms'])
        
        print(f"\n🏆 BEST BATCH SIZE: {best_batch}")
        print(f"   Best throughput: {batch_results[best_batch]['throughput_per_sec']:.1f} emb/sec")
    
    def memory_usage_analysis(self):
        """Analyze memory usage patterns"""
        
        print(f"\n💾 MEMORY USAGE ANALYSIS")
        print("-" * 30)
        
        import psutil
        import gc
        
        # Get baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024**2
        
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        
        # Test with different text lengths
        text_lengths = [10, 50, 100, 200, 512]  # In tokens (approximately)
        
        for length in text_lengths:
            # Create text of approximate length
            words = ["test", "word", "example", "content"] * (length // 4)
            text = " ".join(words[:length])
            
            # Generate embedding
            _ = self.loader.generate_embedding_cpu(text)
            
            # Measure memory
            current_memory = psutil.Process().memory_info().rss / 1024**2
            memory_increase = current_memory - baseline_memory
            
            print(f"Text length ~{length} tokens: +{memory_increase:.1f} MB")
        
        print(f"Model memory: {self.model_info['memory_gb']*1024:.1f} MB")
    
    def performance_targets(self):
        """Calculate NPU performance targets"""
        
        print(f"\n🎯 NPU PERFORMANCE TARGETS")
        print("=" * 40)
        
        single_baseline = self.results['single_embeddings']['throughput_per_sec']
        batch_baseline = max(self.results['batch_embeddings'][bs]['throughput_per_sec'] 
                           for bs in self.results['batch_embeddings'])
        
        print(f"CPU BASELINE:")
        print(f"   Single: {single_baseline:.1f} embeddings/sec")
        print(f"   Best batch: {batch_baseline:.1f} embeddings/sec")
        
        print(f"\nNPU TARGETS (3-5x speedup):")
        print(f"   Conservative (3x): {single_baseline * 3:.1f} embeddings/sec")
        print(f"   Target (4x): {single_baseline * 4:.1f} embeddings/sec") 
        print(f"   Stretch (5x): {single_baseline * 5:.1f} embeddings/sec")
        
        print(f"\nNPU BATCH TARGETS:")
        print(f"   Conservative (3x): {batch_baseline * 3:.1f} embeddings/sec")
        print(f"   Target (4x): {batch_baseline * 4:.1f} embeddings/sec")
        print(f"   Stretch (5x): {batch_baseline * 5:.1f} embeddings/sec")
        
        # Success criteria
        self.results['npu_targets'] = {
            'single_embeddings': {
                'conservative_3x': single_baseline * 3,
                'target_4x': single_baseline * 4,
                'stretch_5x': single_baseline * 5
            },
            'batch_embeddings': {
                'conservative_3x': batch_baseline * 3,
                'target_4x': batch_baseline * 4,
                'stretch_5x': batch_baseline * 5
            }
        }
    
    def save_results(self, filepath: str = None):
        """Save benchmark results to file"""
        
        if filepath is None:
            filepath = Path(__file__).parent / "cpu_baseline_results.json"
        
        # Add metadata
        self.results['metadata'] = {
            'model_info': self.model_info,
            'timestamp': time.time(),
            'python_version': sys.version,
            'numpy_version': np.__version__
        }
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 RESULTS SAVED: {filepath}")
    
    def run_complete_benchmark(self):
        """Run complete baseline benchmark"""
        
        print("🚀 RUNNING COMPLETE CPU BASELINE BENCHMARK")
        print("=" * 60)
        
        self.setup()
        self.benchmark_single_embeddings()
        self.benchmark_batch_embeddings()
        self.memory_usage_analysis()
        self.performance_targets()
        self.save_results()
        
        print(f"\n✅ BASELINE BENCHMARK COMPLETE!")
        print(f"📊 Results available for NPU optimization targeting")

def main():
    """Run CPU baseline benchmark"""
    
    benchmark = CPUBaselineBenchmark()
    benchmark.run_complete_benchmark()

if __name__ == "__main__":
    main()