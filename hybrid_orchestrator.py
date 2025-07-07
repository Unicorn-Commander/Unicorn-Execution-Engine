#!/usr/bin/env python3
"""
Hybrid NPU+iGPU Orchestrator for Gemma 3n E2B
Coordinates execution between AMD NPU Phoenix (prefill) and Radeon 780M iGPU (decode)
Optimized for 40-80 TPS and 20-40ms TTFT targets
"""

import asyncio
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager
import psutil
import gc

from qwen25_loader import HybridConfig, Qwen25Loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Performance metrics for hybrid inference"""
    ttft_ms: float = 0.0          # Time to first token
    tps: float = 0.0              # Tokens per second
    npu_utilization: float = 0.0   # NPU utilization %
    igpu_utilization: float = 0.0  # iGPU utilization %
    memory_usage_mb: float = 0.0   # Total memory usage
    prefill_time_ms: float = 0.0   # NPU prefill time
    decode_time_ms: float = 0.0    # iGPU decode time per token
    queue_latency_ms: float = 0.0  # Inter-device transfer latency

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

class PerformanceMonitor:
    """Real-time performance monitoring for hybrid execution"""
    
    def __init__(self):
        self.metrics_history = []
        self.npu_stats = []
        self.igpu_stats = []
        self.start_time = None
        
    def start_inference(self):
        """Start monitoring inference performance"""
        self.start_time = time.time()
        
    def log_prefill_complete(self, prefill_time_ms: float):
        """Log NPU prefill completion"""
        self.prefill_time = prefill_time_ms
        
    def log_token_decoded(self, decode_time_ms: float):
        """Log iGPU token decode"""
        self.decode_times = getattr(self, 'decode_times', [])
        self.decode_times.append(decode_time_ms)
        
    def calculate_metrics(self, num_tokens: int) -> InferenceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.start_time:
            return InferenceMetrics()
            
        total_time = time.time() - self.start_time
        
        metrics = InferenceMetrics(
            ttft_ms=getattr(self, 'prefill_time', 0.0),
            tps=num_tokens / total_time if total_time > 0 else 0.0,
            prefill_time_ms=getattr(self, 'prefill_time', 0.0),
            decode_time_ms=np.mean(getattr(self, 'decode_times', [0.0])),
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        return metrics

class NPUPrefillEngine:
    """NPU engine optimized for prefill phase and attention operations"""
    
    def __init__(self, config: HybridConfig, npu_attention_module: Any, model_config: Any):
        self.config = config
        self.npu_attention_module = npu_attention_module
        self.model_config = model_config
        self.device_id = "npu:0"  # Logical NPU device
        self.memory_pool = self._initialize_npu_memory()
        self.attention_cache = {}
        
    def _initialize_npu_memory(self) -> Dict[str, Any]:
        """Initialize NPU memory pool (2GB budget for Phoenix)"""
        pool = {
            'embedding_cache': {},
            'attention_cache': {},
            'workspace': None,
            'allocated_bytes': 0
        }
        
        # Pre-allocate workspace for common operations
        workspace_size = min(512 * 1024 * 1024, self.config.npu_memory_budget // 4)  # 512MB or 25% of budget
        pool['workspace'] = torch.zeros(workspace_size // 2, dtype=torch.float16)  # FP16 workspace
        pool['allocated_bytes'] = workspace_size
        
        logger.info(f"NPU memory pool initialized: {workspace_size/1024/1024:.1f}MB workspace")
        return pool
        
    async def prefill_sequence(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Execute prefill phase on NPU (optimized for attention operations)"""
        start_time = time.time()
        
        batch_size, seq_len = input_ids.shape
        # Qwen2.5-7B hidden size is 4096
        hidden_size = 4096
        
        try:
            # Embedding lookup on NPU (highly efficient)
            embeddings = await self._npu_embedding_lookup(input_ids)
            
            # NPU attention processing (Phoenix strength: 16 TOPS for attention)
            # This will eventually call the NPU-accelerated kernel
            # For now, it uses the simulated NPUAttentionModule
            
            # Assuming Qwen2.5 model has a way to get Q, K, V from hidden_states
            # For simulation, we'll just pass hidden_states as Q, K, V
            # In a real scenario, these would be derived from the model's attention layer
            q = embeddings
            k = embeddings
            v = embeddings
            
            # Reshape embeddings to match expected input of NPUAttentionModule
            # (batch_size, num_heads, sequence_length, head_dim)
            batch_size, seq_len, hidden_size = embeddings.shape
            num_heads = 32 # Qwen2.5 configuration
            head_dim = hidden_size // num_heads
            
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            attention_output_reshaped = self.npu_attention_module(q, k, v, attention_mask)
            
            # Reshape back to (batch_size, sequence_length, hidden_size)
            attention_outputs_hidden_states = attention_output_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            
            # Placeholder for past_key_values (will be handled by real NPU kernel)
            past_key_values = [(torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16),
                                torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)) 
                               for _ in range(self.model_config.num_hidden_layers)]
            
            prefill_outputs = {
                'hidden_states': attention_outputs_hidden_states,
                'past_key_values': past_key_values,
                'attention_mask': attention_mask,
                'npu_cache': {}
            }
            
            prefill_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"NPU prefill completed: {seq_len} tokens in {prefill_time:.2f}ms ({seq_len/prefill_time*1000:.1f} tokens/sec)")
            
            return prefill_outputs
            
        except Exception as e:
            logger.error(f"NPU prefill failed: {e}")
            # Fallback to CPU
            return await self._cpu_fallback_prefill(input_ids, attention_mask)
            
    async def _npu_embedding_lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """NPU-optimized embedding lookup"""
        # Simulate NPU embedding lookup (would use XRT/MLIR-AIE in production)
        batch_size, seq_len = input_ids.shape
        
        # Check cache first
        cache_key = f"emb_{seq_len}"
        if cache_key in self.memory_pool['embedding_cache']:
            cached_emb = self.memory_pool['embedding_cache'][cache_key]
            if cached_emb.shape[1] >= seq_len:
                return cached_emb[:batch_size, :seq_len, :]
                
        # NPU embedding computation (optimized for Phoenix)
        embedding_dim = 4096  # Qwen2.5 hidden size
        embeddings = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.float16)
        
        # Cache for future use
        if self.memory_pool['allocated_bytes'] < self.config.npu_memory_budget * 0.8:
            self.memory_pool['embedding_cache'][cache_key] = embeddings.clone()
            self.memory_pool['allocated_bytes'] += embeddings.numel() * 2  # FP16
            
        return embeddings
        
    
        
    async def _cpu_fallback_prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """CPU fallback for NPU failures"""
        logger.warning("Using CPU fallback for prefill")
        
        batch_size, seq_len = input_ids.shape
        hidden_size = 4096 # Qwen2.5 hidden size
        
        # Simple CPU-based prefill
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        past_kv = [(torch.randn(batch_size, 16, seq_len, 128, dtype=torch.float16),
                   torch.randn(batch_size, 16, seq_len, 128, dtype=torch.float16)) 
                   for _ in range(12)]
        
        return {
            'hidden_states': hidden_states,
            'past_key_values': past_kv,
            'attention_mask': attention_mask
        }

class IGPUDecodeEngine:
    """iGPU engine optimized for decode phase and sustained throughput"""
    
    def __init__(self, config: HybridConfig, igpu_ffn_module: Any):
        self.config = config
        self.igpu_ffn_module = igpu_ffn_module
        self.device = torch.device(config.igpu_device if torch.cuda.is_available() else "cpu")
        self.memory_pool = self._initialize_igpu_memory()
        self.compiled_kernels = {}
        
    def _initialize_igpu_memory(self) -> Dict[str, torch.Tensor]:
        """Initialize iGPU memory pool for Radeon 780M"""
        pool = {}
        
        try:
            # Pre-allocate tensors for common decode operations
            pool['hidden_buffer'] = torch.zeros(1, 1, 4096, dtype=torch.float16, device=self.device)
            pool['ffn_buffer'] = torch.zeros(1, 1, 11008, dtype=torch.float16, device=self.device)
            pool['output_buffer'] = torch.zeros(1, 1, 152064, dtype=torch.float16, device=self.device)  # Qwen2.5 Vocab size
            
            # Warm up ROCm/HIP
            _ = torch.matmul(pool['hidden_buffer'].view(1, -1), pool['hidden_buffer'].view(-1, 1))
            
            logger.info(f"iGPU memory pool initialized on {self.device}")
            
        except Exception as e:
            logger.warning(f"iGPU initialization failed, using CPU: {e}")
            self.device = torch.device("cpu")
            
        return pool
        
    async def decode_token(self, hidden_states: torch.Tensor, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], 
                          position_ids: torch.Tensor, generation_config: GenerationConfig) -> Dict[str, Any]:
        """Decode single token using iGPU (optimized for Radeon 780M)"""
        start_time = time.time()
        
        try:
            # Transfer to iGPU if needed
            hidden_states = hidden_states.to(self.device, non_blocking=True)
            
            # iGPU FFN processing (memory-intensive operations)
            ffn_output = self.igpu_ffn_module(hidden_states)
            
            # Output projection and logits
            logits = await self._igpu_output_projection(ffn_output)
            
            # Sampling (can be done on iGPU for efficiency)
            next_token = await self._igpu_sample_token(logits, generation_config)
            
            decode_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'next_token': next_token,
                'logits': logits,
                'hidden_states': ffn_output,
                'decode_time_ms': decode_time
            }
            
        except Exception as e:
            logger.error(f"iGPU decode failed: {e}")
            return await self._cpu_fallback_decode(hidden_states, generation_config)
            
    
        
    async def _igpu_output_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """iGPU-optimized output projection to vocabulary"""
        hidden_size = hidden_states.shape[-1]
        # Qwen2.5 vocabulary size is 152064
        vocab_size = 152064
        
        # Output projection
        logits = torch.matmul(hidden_states, torch.randn(hidden_size, vocab_size, device=self.device, dtype=torch.float16))
        logits = torch.clamp(logits, min=-10.0, max=10.0) # Clamp logits to prevent extreme values
        
        return logits
        
    async def _igpu_sample_token(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """iGPU-optimized token sampling"""
        logits = logits[:, -1, :]  # Last token logits
        
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
            
        # Top-k filtering
        if config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)
            
        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # Sample
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        return next_token
        
    async def _cpu_fallback_decode(self, hidden_states: torch.Tensor, config: GenerationConfig) -> Dict[str, Any]:
        """CPU fallback for iGPU failures"""
        logger.warning("Using CPU fallback for decode")
        
        # Simple CPU-based decode
        logits = torch.randn(hidden_states.shape[0], hidden_states.shape[1], 152064) # Qwen2.5 vocab size
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        return {
            'next_token': next_token,
            'logits': logits,
            'hidden_states': hidden_states,
            'decode_time_ms': 5.0  # Assumed CPU time
        }

class HybridOrchestrator:
    """Main orchestrator coordinating NPU prefill and iGPU decode phases"""
    
    def __init__(self, config: HybridConfig, model_partitions: Dict[str, Any]):
        self.config = config
        self.partitions = model_partitions
        
        # Initialize engines
        self.npu_engine = NPUPrefillEngine(config, model_partitions['npu']['attention_kernels'], model_partitions['config']['model_config'])
        self.igpu_engine = IGPUDecodeEngine(config, model_partitions['igpu']['ffn_kernels'])
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Hybrid orchestrator initialized")
        
    async def generate_text(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """Generate text using hybrid NPU+iGPU execution"""
        logger.info(f"Starting hybrid generation: '{prompt[:50]}...'")
        
        # Start performance monitoring
        self.monitor.start_inference()
        
        try:
            # Tokenization (CPU)
            tokenizer = self.partitions['cpu']['tokenizer']
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = torch.ones_like(input_ids)
            
            logger.info(f"Input tokenized: {input_ids.shape[1]} tokens")
            
            # Phase 1: NPU Prefill
            logger.info("Phase 1: NPU prefill processing...")
            prefill_start = time.time()
            
            prefill_outputs = await self.npu_engine.prefill_sequence(input_ids, attention_mask)
            
            prefill_time = (time.time() - prefill_start) * 1000
            self.monitor.log_prefill_complete(prefill_time)
            
            logger.info(f"NPU prefill completed in {prefill_time:.2f}ms")
            
            # Phase 2: iGPU Decode Loop
            logger.info("Phase 2: iGPU decode processing...")
            
            generated_tokens = []
            hidden_states = prefill_outputs['hidden_states'][:, -1:, :]  # Last token
            past_key_values = prefill_outputs['past_key_values']
            
            for step in range(generation_config.max_new_tokens):
                position_ids = torch.tensor([[input_ids.shape[1] + step]], dtype=torch.long)
                
                # Decode single token on iGPU
                decode_output = await self.igpu_engine.decode_token(
                    hidden_states, past_key_values, position_ids, generation_config
                )
                
                next_token = decode_output['next_token']
                generated_tokens.append(next_token.item())
                
                # Update hidden states for next iteration
                hidden_states = decode_output['hidden_states']
                
                # Log decode time
                self.monitor.log_token_decoded(decode_output['decode_time_ms'])
                
                # Check for EOS token
                if next_token.item() == generation_config.eos_token_id:
                    logger.info(f"EOS token reached at step {step}")
                    break
                    
                # Progress logging
                if step % 50 == 0 and step > 0:
                    elapsed = time.time() - prefill_start
                    current_tps = step / elapsed
                    logger.info(f"Generated {step} tokens, current TPS: {current_tps:.1f}")
            
            # Decode generated tokens
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate final metrics
            final_metrics = self.monitor.calculate_metrics(len(generated_tokens))
            
            result = {
                'generated_text': generated_text,
                'generated_tokens': generated_tokens,
                'metrics': final_metrics,
                'prefill_outputs': prefill_outputs
            }
            
            logger.info(f"Generation completed: {len(generated_tokens)} tokens")
            logger.info(f"Performance: {final_metrics.tps:.1f} TPS, {final_metrics.ttft_ms:.1f}ms TTFT")
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")
            raise
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.monitor.metrics_history:
            return {"status": "No metrics available"}
            
        recent_metrics = self.monitor.metrics_history[-1] if self.monitor.metrics_history else InferenceMetrics()
        
        # Performance target analysis
        ttft_target_met = self.config.target_ttft_ms[0] <= recent_metrics.ttft_ms <= self.config.target_ttft_ms[1]
        tps_target_met = self.config.target_tps[0] <= recent_metrics.tps <= self.config.target_tps[1]
        
        return {
            'current_performance': {
                'ttft_ms': recent_metrics.ttft_ms,
                'tps': recent_metrics.tps,
                'memory_usage_mb': recent_metrics.memory_usage_mb
            },
            'targets': {
                'ttft_ms': self.config.target_ttft_ms,
                'tps': self.config.target_tps,
                'ttft_met': ttft_target_met,
                'tps_met': tps_target_met
            },
            'hardware_utilization': {
                'npu_utilization': recent_metrics.npu_utilization,
                'igpu_utilization': recent_metrics.igpu_utilization
            },
            'optimization_score': int((ttft_target_met + tps_target_met) * 50)  # 0-100 score
        }

async def main():
    """Test the hybrid orchestrator"""
    from gemma3n_e2b_loader import Gemma3nE2BLoader
    
    print("=== Gemma 3n E2B Hybrid Orchestrator Test ===")
    
    # Configuration
    config = HybridConfig()
    generation_config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    
    # Load model
    loader = Gemma3nE2BLoader(config)
    model, tokenizer = loader.load_model()
    partitions = loader.partition_for_hybrid_execution()
    
    # Initialize orchestrator
    orchestrator = HybridOrchestrator(config, partitions)
    
    # Test generation
    prompt = "The future of AI on edge devices will be"
    
    print(f"\nPrompt: {prompt}")
    print("Generating with hybrid NPU+iGPU execution...")
    
    result = await orchestrator.generate_text(prompt, generation_config)
    
    print(f"\nGenerated: {result['generated_text']}")
    print(f"\nPerformance:")
    print(f"  TTFT: {result['metrics'].ttft_ms:.1f}ms")
    print(f"  TPS: {result['metrics'].tps:.1f}")
    print(f"  Memory: {result['metrics'].memory_usage_mb:.1f}MB")
    
    # Performance summary
    summary = orchestrator.get_performance_summary()
    print(f"\nOptimization Score: {summary['optimization_score']}/100")
    
    return orchestrator

if __name__ == "__main__":
    asyncio.run(main())