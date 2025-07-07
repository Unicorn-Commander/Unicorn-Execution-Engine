# Gemma 3B NPU/iGPU Hybrid Implementation Project

## System Configuration
- **CPU**: AMD Ryzen 9045HS (Phoenix Point)
- **iGPU**: AMD Radeon 780M (12 CUs, RDNA3)
- **NPU**: AMD XDNA AI Engine (16 TOPS)
- **RAM**: 96GB total (16GB allocated as VRAM)
- **OS**: Ubuntu Server 25.04, Kernel 6.14, KDE5

## Project Overview

Implement Gemma 3B (2B active parameters) with intelligent workload distribution across NPU, iGPU, and CPU for optimal performance and efficiency.

## Architecture Design

### 1. Compute Unit Specialization
```
NPU (16 TOPS):
- Matrix multiplications (attention K/Q/V projections)
- Embedding lookups
- Small convolutions
- Activation functions (GELU, softmax components)

iGPU (780M, ~8.6 TFLOPS FP16):
- Large attention computations
- FFN layers
- Layer normalization
- Residual connections

CPU (8 cores, 16 threads):
- Tokenization
- Sampling/beam search
- Memory management
- Orchestration
```

### 2. Memory Architecture
```
System RAM (80GB available):
- Model weights (original)
- KV cache
- Intermediate buffers

VRAM (16GB):
- Active layer weights
- Current attention states
- iGPU compute buffers

NPU Memory (4GB estimated):
- Quantized weight tiles
- Activation caches
- Kernel buffers
```

## Implementation Components

### 1. NPU Kernel Library (`npu_kernels.py`)
```python
import numpy as np
from mlir_aie import MLIRAIECompiler
from xrt import XRTDevice

class GemmaNPUKernels:
    def __init__(self):
        self.device = XRTDevice()
        self.compiler = MLIRAIECompiler()
        
    def compile_attention_kernel(self):
        """Compile attention projection kernels for NPU"""
        kernel = """
        func @attention_qkv(%input: tensor<2048x3072xf16>,
                           %w_q: tensor<3072x1024xf16>,
                           %w_k: tensor<3072x1024xf16>,
                           %w_v: tensor<3072x1024xf16>)
                           -> (tensor<2048x1024xf16>, 
                               tensor<2048x1024xf16>,
                               tensor<2048x1024xf16>) {
            %q = linalg.matmul ins(%input, %w_q)
            %k = linalg.matmul ins(%input, %w_k)
            %v = linalg.matmul ins(%input, %w_v)
            return %q, %k, %v
        }
        """
        return self.compiler.compile(kernel, target="npu")
    
    def compile_embedding_kernel(self):
        """Compile embedding lookup for NPU"""
        kernel = """
        func @embedding_lookup(%indices: tensor<2048xi32>,
                              %table: tensor<256128x3072xf16>)
                              -> tensor<2048x3072xf16> {
            %result = tensor.gather %table[%indices, :]
            return %result
        }
        """
        return self.compiler.compile(kernel, target="npu")
```

### 2. iGPU Backend (`igpu_backend.py`)
```python
import torch
import torch_mlir
from torch.utils.dlpack import to_dlpack, from_dlpack

class GemmaIGPUBackend:
    def __init__(self, vram_gb=16):
        self.device = torch.device("cuda")  # ROCm/HIP backend
        self.vram_budget = vram_gb * 1024**3
        
    def compile_attention_layer(self, config):
        """Compile full attention mechanism for iGPU"""
        class AttentionLayer(torch.nn.Module):
            def __init__(self, dim=3072, heads=16):
                super().__init__()
                self.dim = dim
                self.heads = heads
                self.head_dim = dim // heads
                
            def forward(self, q, k, v, mask=None):
                # Reshape for multi-head attention
                B, L, _ = q.shape
                q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
                k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
                v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                if mask is not None:
                    scores.masked_fill_(mask == 0, -1e9)
                    
                attn_weights = torch.nn.functional.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
                
                return output.transpose(1, 2).contiguous().view(B, L, -1)
        
        return torch.jit.script(AttentionLayer(config.hidden_size, config.num_heads))
    
    def compile_ffn_layer(self, config):
        """Compile feed-forward network for iGPU"""
        class FFNLayer(torch.nn.Module):
            def __init__(self, dim=3072, hidden_dim=8192):
                super().__init__()
                self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
                self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
                self.act = torch.nn.GELU()
                
            def forward(self, x):
                return self.w2(self.act(self.w1(x)))
        
        return torch.jit.script(FFNLayer(config.hidden_size, config.intermediate_size))
```

### 3. Hybrid Orchestrator (`hybrid_orchestrator.py`)
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class GemmaHybridOrchestrator:
    def __init__(self, npu_backend, igpu_backend):
        self.npu = npu_backend
        self.igpu = igpu_backend
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def forward_layer(self, layer_idx, hidden_states, past_kv=None):
        """Execute one transformer layer with hybrid compute"""
        
        # Step 1: NPU computes Q,K,V projections
        qkv_future = self.executor.submit(
            self.npu.compute_qkv_projections,
            hidden_states,
            layer_idx
        )
        
        # Step 2: While NPU is busy, prepare attention mask on CPU
        attention_mask = self.prepare_attention_mask(hidden_states.shape)
        
        # Step 3: Get NPU results and transfer to iGPU
        q, k, v = await asyncio.wrap_future(qkv_future)
        q_gpu = self.transfer_to_igpu(q)
        k_gpu = self.transfer_to_igpu(k)
        v_gpu = self.transfer_to_igpu(v)
        
        # Step 4: iGPU computes full attention
        attn_output = await self.igpu.compute_attention_async(
            q_gpu, k_gpu, v_gpu, attention_mask
        )
        
        # Step 5: FFN on iGPU while NPU prepares next layer
        ffn_output = await self.igpu.compute_ffn_async(attn_output)
        
        return ffn_output, (k, v)  # Return KV for caching
    
    def transfer_to_igpu(self, npu_tensor):
        """Efficient NPU->iGPU transfer via system RAM"""
        # Use pinned memory for faster transfers
        pinned = np.asarray(npu_tensor, order='C')
        return torch.from_numpy(pinned).to(self.igpu.device, non_blocking=True)
```

### 4. Model Loader and Quantizer (`model_loader.py`)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnx

class GemmaModelLoader:
    def __init__(self, model_id="google/gemma-2-2b"):
        self.model_id = model_id
        
    def load_and_partition(self):
        """Load Gemma and partition for hybrid execution"""
        # Load original model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Extract and partition components
        partitions = {
            'npu': {
                'embeddings': model.model.embed_tokens,
                'projections': [],
                'activations': []
            },
            'igpu': {
                'attention_layers': [],
                'ffn_layers': [],
                'norm_layers': []
            },
            'cpu': {
                'lm_head': model.lm_head,
                'tokenizer': AutoTokenizer.from_pretrained(self.model_id)
            }
        }
        
        # Partition transformer layers
        for idx, layer in enumerate(model.model.layers):
            # NPU gets projections
            partitions['npu']['projections'].append({
                'q_proj': self.quantize_for_npu(layer.self_attn.q_proj),
                'k_proj': self.quantize_for_npu(layer.self_attn.k_proj),
                'v_proj': self.quantize_for_npu(layer.self_attn.v_proj)
            })
            
            # iGPU gets attention and FFN
            partitions['igpu']['attention_layers'].append(
                self.prepare_for_igpu(layer.self_attn)
            )
            partitions['igpu']['ffn_layers'].append(
                self.prepare_for_igpu(layer.mlp)
            )
            
        return partitions
    
    def quantize_for_npu(self, module):
        """Quantize weights for NPU (INT8/FP16)"""
        weights = module.weight.data
        # Simple symmetric quantization to FP16
        return weights.half().numpy()
```

### 5. Inference Pipeline (`inference_pipeline.py`)
```python
import time
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    max_length: int = 2048
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
class GemmaInferencePipeline:
    def __init__(self, orchestrator, model_partitions, config):
        self.orchestrator = orchestrator
        self.model = model_partitions
        self.config = config
        self.kv_cache = {}
        
    async def generate(self, prompt, max_new_tokens=512):
        """Generate text using hybrid NPU/iGPU execution"""
        # Tokenize on CPU
        input_ids = self.model['cpu']['tokenizer'].encode(prompt)
        
        # Embedding lookup on NPU
        hidden_states = await self.orchestrator.npu.embedding_lookup(input_ids)
        
        # Process through transformer layers
        for layer_idx in range(self.model['num_layers']):
            hidden_states, new_kv = await self.orchestrator.forward_layer(
                layer_idx, 
                hidden_states,
                self.kv_cache.get(layer_idx)
            )
            self.kv_cache[layer_idx] = new_kv
            
        # Final LM head on CPU for flexibility
        logits = self.model['cpu']['lm_head'](hidden_states)
        
        # Sample next token
        next_token = self.sample_token(logits)
        
        return next_token
    
    def sample_token(self, logits):
        """Advanced sampling with temperature, top-k, top-p"""
        # Apply temperature
        logits = logits / self.config.temperature
        
        # Top-k filtering
        if self.config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
            
        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        # Sample from distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
```

### 6. Performance Monitor (`performance_monitor.py`)
```python
import psutil
import pynvml  # For AMD GPUs via ROCm-SMI
from collections import deque

class HybridPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'npu_utilization': deque(maxlen=100),
            'igpu_utilization': deque(maxlen=100),
            'cpu_utilization': deque(maxlen=100),
            'memory_bandwidth': deque(maxlen=100),
            'tokens_per_second': deque(maxlen=100)
        }
        
    def log_inference_step(self, step_metrics):
        """Log metrics for one inference step"""
        # NPU metrics via XRT-SMI
        npu_util = self.get_npu_utilization()
        self.metrics['npu_utilization'].append(npu_util)
        
        # iGPU metrics via ROCm-SMI
        igpu_util = self.get_igpu_utilization()
        self.metrics['igpu_utilization'].append(igpu_util)
        
        # System metrics
        self.metrics['cpu_utilization'].append(psutil.cpu_percent())
        self.metrics['memory_bandwidth'].append(self.get_memory_bandwidth())
        self.metrics['tokens_per_second'].append(step_metrics['tps'])
        
    def get_optimization_suggestions(self):
        """Analyze metrics and suggest optimizations"""
        suggestions = []
        
        # Check NPU utilization
        avg_npu = np.mean(self.metrics['npu_utilization'])
        if avg_npu < 70:
            suggestions.append("NPU underutilized - consider moving more ops")
            
        # Check iGPU utilization
        avg_igpu = np.mean(self.metrics['igpu_utilization'])
        if avg_igpu > 90:
            suggestions.append("iGPU saturated - offload some ops to NPU")
            
        # Check memory bandwidth
        avg_bw = np.mean(self.metrics['memory_bandwidth'])
        if avg_bw > 80:
            suggestions.append("Memory bottleneck - implement tensor fusion")
            
        return suggestions
```

### 7. Main Execution Script (`run_gemma_hybrid.py`)
```python
#!/usr/bin/env python3
import asyncio
import argparse
from pathlib import Path

async def main():
    parser = argparse.ArgumentParser(description='Run Gemma 3B with NPU/iGPU hybrid execution')
    parser.add_argument('--model', default='google/gemma-2-2b', help='Model ID')
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    args = parser.parse_args()
    
    print("Initializing Gemma hybrid execution environment...")
    
    # Initialize backends
    npu_backend = GemmaNPUKernels()
    igpu_backend = GemmaIGPUBackend(vram_gb=16)
    
    # Create orchestrator
    orchestrator = GemmaHybridOrchestrator(npu_backend, igpu_backend)
    
    # Load and partition model
    print("Loading and partitioning Gemma model...")
    loader = GemmaModelLoader(args.model)
    model_partitions = loader.load_and_partition()
    
    # Create inference pipeline
    config = InferenceConfig()
    pipeline = GemmaInferencePipeline(orchestrator, model_partitions, config)
    
    # Initialize performance monitor
    monitor = HybridPerformanceMonitor()
    
    # Run inference
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating response...\n")
    
    start_time = time.time()
    response = await pipeline.generate(args.prompt, max_new_tokens=args.max_tokens)
    end_time = time.time()
    
    print(f"\nResponse: {response}")
    print(f"\nGeneration time: {end_time - start_time:.2f}s")
    print(f"Tokens per second: {args.max_tokens / (end_time - start_time):.2f}")
    
    # Show optimization suggestions
    suggestions = monitor.get_optimization_suggestions()
    if suggestions:
        print("\nOptimization suggestions:")
        for s in suggestions:
            print(f"  - {s}")
    
    if args.benchmark:
        print("\nRunning benchmark suite...")
        await run_benchmark_suite(pipeline, monitor)

if __name__ == "__main__":
    asyncio.run(main())
```

## Installation Instructions

### 1. Setup NPU Development Environment
```bash
# Assuming NPU toolkit is already installed from your previous work
source ~/npu-dev/setup_npu_env.sh

# Install additional dependencies
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install onnx onnxruntime torch-mlir
pip install asyncio aiofiles psutil py3nvml
```

### 2. Install ROCm for iGPU Support
```bash
# Add AMD ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0.2 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dkms rocm-libs miopen-hip

# Add user to render and video groups
sudo usermod -a -G render,video $USER

# Set environment variables
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### 3. Verify Setup
```bash
# Check NPU
xrt-smi examine

# Check iGPU
rocm-smi

# Should show AMD Radeon 780M
```

## Performance Optimization Tips

### 1. Memory Management
- Use pinned memory for CPU->GPU transfers
- Implement double buffering for KV cache
- Pre-allocate all buffers to avoid runtime allocation

### 2. Kernel Fusion
- Fuse embedding + position encoding on NPU
- Combine layer norm + residual on iGPU
- Batch small operations to reduce kernel launch overhead

### 3. Dynamic Load Balancing
- Monitor thermal throttling
- Shift load from throttled units
- Use power profiles: balanced, performance, efficiency

### 4. Quantization Strategy
- FP16 for iGPU operations
- INT8 for NPU matrix multiplies
- Keep accumulations in FP32 for accuracy

## Expected Performance

With optimal configuration:
- **First token latency**: ~50-100ms
- **Throughput**: 30-50 tokens/second
- **Power efficiency**: ~0.5W per token
- **Memory usage**: ~8GB active, 4GB cached

## Troubleshooting

### NPU Issues
```bash
# Check NPU driver
lsmod | grep amdxdna

# Reset NPU
sudo rmmod amdxdna
sudo modprobe amdxdna

# Verify NPU memory
xrt-smi examine -r platform
```

### iGPU Issues
```bash
# Check ROCm installation
rocminfo

# Monitor GPU
rocm-smi --showuse

# Reset GPU
sudo rocm-smi --gpu_reset 0
```

### Performance Issues
1. Check thermal throttling: `sensors`
2. Monitor memory bandwidth: `sudo perf stat -e cache-misses`
3. Profile kernel execution: `rocprof python run_gemma_hybrid.py`

## Next Steps

1. Implement the complete codebase
2. Run initial benchmarks
3. Profile and identify bottlenecks
4. Implement advanced optimizations:
   - Custom MLIR kernels for specific patterns
   - Vulkan compute shaders for additional parallelism
   - Dynamic batching for improved throughput
   - Speculative decoding with small model on NPU

This architecture maximizes your hardware:
- NPU handles high-frequency, small operations
- iGPU processes bulk compute
- CPU orchestrates and handles complex logic
- 96GB RAM enables large batch sizes and extensive caching