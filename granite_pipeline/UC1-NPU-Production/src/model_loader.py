#!/usr/bin/env python3
"""
Qwen3-Embedding-0.6B Model Loader for UC1-NPU-Production
Loads real Qwen3 embedding model and provides CPU baseline
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import time
import sys
import os

class Qwen3EmbeddingLoader:
    """Load and manage Qwen3-Embedding-0.6B model"""
    
    def __init__(self, model_path: str = None):
        # Use the HuggingFace model ID directly
        self.model_path = model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")  # Start with CPU baseline
        
    def download_model(self):
        """Download Qwen3-Embedding-0.6B if not present"""
        
        # Since we're using HuggingFace model ID, it will auto-download
        print(f"📥 Using model: {self.model_path}")
        return True
    
    def load_model(self):
        """Load the Qwen3 model and tokenizer"""
        
        if not self.download_model():
            return False
            
        try:
            print(f"🔄 Loading Qwen3-Embedding-0.6B...")
            
            if self.model is None:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Memory: {sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_embedding_cpu(self, text: str) -> np.ndarray:
        """Generate embedding using CPU (baseline)"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling (or CLS token for some models)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                # Mean pooling across sequence dimension
                embedding = hidden_states.mean(dim=1)
            else:
                # Some models return embeddings directly
                embedding = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0].mean(dim=1)
        
        # Normalize embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def benchmark_cpu_performance(self, test_texts: list = None, num_runs: int = 10):
        """Benchmark CPU embedding performance"""
        
        if test_texts is None:
            test_texts = [
                "The quick brown fox jumps over the lazy dog",
                "Artificial intelligence is transforming the world",
                "Natural language processing enables human-computer interaction",
                "Machine learning algorithms learn patterns from data",
                "Deep neural networks can model complex relationships"
            ]
        
        print(f"\n📊 BENCHMARKING CPU PERFORMANCE")
        print(f"=" * 50)
        print(f"Test texts: {len(test_texts)}")
        print(f"Runs per text: {num_runs}")
        
        all_times = []
        
        # Warm-up
        self.generate_embedding_cpu(test_texts[0])
        
        for i, text in enumerate(test_texts):
            times = []
            
            for run in range(num_runs):
                start_time = time.perf_counter()
                embedding = self.generate_embedding_cpu(text)
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                
                if run == 0:
                    print(f"Text {i+1}: {text[:50]}...")
                    print(f"   Embedding shape: {embedding.shape}")
                    print(f"   First run: {elapsed_ms:.2f}ms")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            all_times.extend(times)
            
            print(f"   Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
            print(f"   Throughput: {1000/avg_time:.1f} embeddings/sec")
        
        # Overall statistics
        overall_avg = np.mean(all_times)
        overall_std = np.std(all_times)
        
        print(f"\n🎯 OVERALL CPU BASELINE:")
        print(f"   Average latency: {overall_avg:.2f}ms")
        print(f"   Std deviation: {overall_std:.2f}ms")
        print(f"   Throughput: {1000/overall_avg:.1f} embeddings/sec")
        print(f"   Total tests: {len(all_times)}")
        
        return {
            'avg_latency_ms': overall_avg,
            'std_latency_ms': overall_std,
            'throughput_per_sec': 1000/overall_avg,
            'all_times_ms': all_times
        }
    
    def get_model_info(self):
        """Get detailed model information"""
        
        if self.model is None:
            return None
            
        config = self.model.config
        
        return {
            'model_name': 'Qwen3-Embedding-0.6B',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.max_position_embeddings,
            'memory_gb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
        }

def main():
    """Test the Qwen3 model loader"""
    
    print("🚀 QWEN3-EMBEDDING-0.6B MODEL LOADER TEST")
    print("=" * 60)
    
    # Initialize loader
    loader = Qwen3EmbeddingLoader()
    
    # Load model
    if not loader.load_model():
        print("❌ Failed to load model")
        return
    
    # Show model info
    info = loader.get_model_info()
    if info:
        print(f"\n📋 MODEL INFORMATION:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            elif isinstance(value, int):
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")
    
    # Test single embedding
    test_text = "This is a test sentence for embedding generation"
    print(f"\n🧪 SINGLE EMBEDDING TEST:")
    print(f"   Text: {test_text}")
    
    start_time = time.perf_counter()
    embedding = loader.generate_embedding_cpu(test_text)
    end_time = time.perf_counter()
    
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"   First 10 values: {embedding[:10]}")
    
    # Benchmark performance
    baseline_results = loader.benchmark_cpu_performance()
    
    print(f"\n✅ BASELINE ESTABLISHED!")
    print(f"   Target: 3-5x speedup with NPU")
    print(f"   Current: {baseline_results['throughput_per_sec']:.1f} emb/sec")
    print(f"   NPU Goal: {baseline_results['throughput_per_sec'] * 3:.1f}-{baseline_results['throughput_per_sec'] * 5:.1f} emb/sec")

if __name__ == "__main__":
    main()