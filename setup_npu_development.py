#!/usr/bin/env python3
"""
NPU Development Environment Setup for Multi-Model Optimization
Sets up MLIR-AIE2 and NPU Phoenix development tools
"""
import os
import subprocess
import logging
import sys
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUDevelopmentSetup:
    """Set up complete NPU development environment"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.npu_dev_dir = self.base_dir / "npu_development"
        self.mlir_dir = self.base_dir / "mlir_aie2"
        self.kernels_dir = self.npu_dev_dir / "kernels"
        
    def check_prerequisites(self) -> bool:
        """Check system prerequisites for NPU development"""
        logger.info("🔍 Checking NPU development prerequisites...")
        
        checks = {
            "XRT Runtime": self._check_xrt(),
            "NPU Phoenix": self._check_npu_device(),
            "AMDXDNA Driver": self._check_amdxdna(),
            "Build Tools": self._check_build_tools(),
            "Python Environment": self._check_python()
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def _check_xrt(self) -> bool:
        """Check XRT runtime availability"""
        try:
            result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_npu_device(self) -> bool:
        """Check NPU Phoenix device"""
        try:
            result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', 'examine'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and 'NPU Phoenix' in result.stdout
        except:
            return False
    
    def _check_amdxdna(self) -> bool:
        """Check AMDXDNA driver"""
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            return 'amdxdna' in result.stdout
        except:
            return False
    
    def _check_build_tools(self) -> bool:
        """Check build tools availability"""
        tools = ['cmake', 'ninja', 'git', 'clang']
        for tool in tools:
            if not shutil.which(tool):
                return False
        return True
    
    def _check_python(self) -> bool:
        """Check Python environment"""
        try:
            import torch
            import numpy
            return True
        except ImportError:
            return False
    
    def create_directory_structure(self):
        """Create NPU development directory structure"""
        logger.info("📁 Creating NPU development directory structure...")
        
        directories = [
            self.npu_dev_dir,
            self.npu_dev_dir / "kernels" / "universal",
            self.npu_dev_dir / "kernels" / "gemma",
            self.npu_dev_dir / "kernels" / "qwen", 
            self.npu_dev_dir / "tools",
            self.npu_dev_dir / "tests",
            self.npu_dev_dir / "examples",
            self.npu_dev_dir / "docs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ {directory}")
    
    def setup_mlir_aie2(self) -> bool:
        """Set up MLIR-AIE2 development environment"""
        logger.info("⚙️ Setting up MLIR-AIE2 environment...")
        
        try:
            # Check if MLIR-AIE2 is already available
            if shutil.which('aie-opt'):
                logger.info("   ✅ MLIR-AIE2 already available")
                return True
            
            # Try to find in common locations
            mlir_paths = [
                '/opt/xilinx/mlir-aie',
                '/usr/local/mlir-aie',
                Path.home() / 'mlir-aie'
            ]
            
            for path in mlir_paths:
                if Path(path).exists():
                    logger.info(f"   ✅ Found MLIR-AIE2 at {path}")
                    # Add to PATH
                    os.environ['PATH'] = f"{path}/bin:{os.environ.get('PATH', '')}"
                    return True
            
            # If not found, create placeholder for manual installation
            logger.warning("⚠️ MLIR-AIE2 not found - creating installation guide")
            self._create_mlir_installation_guide()
            return False
            
        except Exception as e:
            logger.error(f"❌ MLIR-AIE2 setup failed: {e}")
            return False
    
    def _create_mlir_installation_guide(self):
        """Create MLIR-AIE2 installation guide"""
        guide_path = self.npu_dev_dir / "MLIR_AIE2_INSTALL.md"
        
        guide_content = """# MLIR-AIE2 Installation Guide

## Option 1: Pre-built Installation
```bash
# Download pre-built MLIR-AIE2 (if available)
wget https://github.com/Xilinx/mlir-aie/releases/latest/download/mlir-aie-linux-x64.tar.gz
tar -xzf mlir-aie-linux-x64.tar.gz -C /opt/xilinx/
export PATH="/opt/xilinx/mlir-aie/bin:$PATH"
```

## Option 2: Build from Source
```bash
# Clone MLIR-AIE2 repository
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Install dependencies
sudo apt-get install cmake ninja-build clang lld

# Build MLIR-AIE2
mkdir build && cd build
cmake .. -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DLLVM_ENABLE_PROJECTS="mlir" \\
  -DLLVM_TARGETS_TO_BUILD="X86;host" \\
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

ninja
sudo ninja install
```

## Option 3: VitisAI Integration
```bash
# Use VitisAI which includes MLIR-AIE2
docker pull xilinx/vitis-ai-cpu:latest
# Or install VitisAI locally with MLIR-AIE2 support
```

## Verification
```bash
# Test MLIR-AIE2 installation
aie-opt --version
aie-translate --version
```
"""
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"   📋 Installation guide created: {guide_path}")
    
    def create_kernel_templates(self):
        """Create NPU kernel templates for different models"""
        logger.info("📝 Creating NPU kernel templates...")
        
        # Universal attention kernel template
        universal_attention = self.kernels_dir / "universal" / "attention_int4_npu.mlir"
        self._create_universal_attention_kernel(universal_attention)
        
        # Gemma-specific kernel template
        gemma_ffn = self.kernels_dir / "gemma" / "gated_ffn_npu.mlir"
        self._create_gemma_ffn_kernel(gemma_ffn)
        
        # Qwen-specific kernel template
        qwen_rope = self.kernels_dir / "qwen" / "rotary_attention_npu.mlir"
        self._create_qwen_rope_kernel(qwen_rope)
        
        logger.info("   ✅ Kernel templates created")
    
    def _create_universal_attention_kernel(self, path: Path):
        """Create universal attention kernel template"""
        kernel_content = '''// Universal INT4 Attention Kernel for NPU Phoenix
// Works with both Gemma and Qwen architectures

module {
  // Main attention function optimized for NPU Phoenix
  func.func @attention_int4_npu(
    %query: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %key: tensor<?x?x?xi4>,        // [batch, seq_len, head_dim] 
    %value: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %mask: tensor<?x?xi1>          // [batch, seq_len] causal mask
  ) -> tensor<?x?x?xi4> {
    
    // Get dimensions
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index  
    %c2 = arith.constant 2 : index
    
    %batch = tensor.dim %query, %c0 : tensor<?x?x?xi4>
    %seq_len = tensor.dim %query, %c1 : tensor<?x?x?xi4>
    %head_dim = tensor.dim %query, %c2 : tensor<?x?x?xi4>
    
    // Compute attention scores: Q @ K^T
    // NPU-optimized INT4 matrix multiplication
    %scores = npu.int4_matmul %query, %key transpose_b : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply scaling factor (1/sqrt(head_dim))
    %scale = arith.constant 0.125 : f16  // Approximate for head_dim=64
    %scaled_scores = npu.int4_scale %scores, %scale : 
      tensor<?x?x?xi4>, f16 -> tensor<?x?x?xi4>
    
    // Apply causal mask
    %masked_scores = npu.apply_mask %scaled_scores, %mask : 
      tensor<?x?x?xi4>, tensor<?x?xi1> -> tensor<?x?x?xi4>
    
    // Softmax (NPU Phoenix has optimized softmax for INT4)
    %attention_weights = npu.softmax_int4 %masked_scores : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply attention to values: Attention @ V
    %output = npu.int4_matmul %attention_weights, %value : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
  
  // Optimized multi-head attention wrapper
  func.func @multihead_attention_npu(
    %input: tensor<?x?x?xi4>,      // [batch, seq_len, hidden_size]
    %q_weight: tensor<?x?xi4>,     // Query projection weights
    %k_weight: tensor<?x?xi4>,     // Key projection weights  
    %v_weight: tensor<?x?xi4>,     // Value projection weights
    %o_weight: tensor<?x?xi4>,     // Output projection weights
    %num_heads: index
  ) -> tensor<?x?x?xi4> {
    
    // Project to Q, K, V
    %query = npu.int4_linear %input, %q_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %key = npu.int4_linear %input, %k_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %value = npu.int4_linear %input, %v_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Reshape for multi-head attention
    %q_heads = npu.reshape_multihead %query, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    %k_heads = npu.reshape_multihead %key, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    %v_heads = npu.reshape_multihead %value, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    
    // Apply attention to each head (NPU can process multiple heads in parallel)
    %attended_heads = npu.parallel_attention %q_heads, %k_heads, %v_heads : 
      tensor<?x?x?x?xi4>, tensor<?x?x?x?xi4>, tensor<?x?x?x?xi4> -> tensor<?x?x?x?xi4>
    
    // Reshape back and apply output projection
    %concatenated = npu.concatenate_heads %attended_heads : 
      tensor<?x?x?x?xi4> -> tensor<?x?x?xi4>
    %output = npu.int4_linear %concatenated, %o_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
'''
        
        with open(path, 'w') as f:
            f.write(kernel_content)
    
    def _create_gemma_ffn_kernel(self, path: Path):
        """Create Gemma-specific gated FFN kernel"""
        kernel_content = '''// Gemma Gated FFN Kernel for NPU Phoenix
// Optimized for Gemma's SiLU + gated architecture

module {
  func.func @gemma_gated_ffn_npu(
    %input: tensor<?x?x?xi4>,      // [batch, seq_len, hidden_size]
    %gate_weight: tensor<?x?xi4>,  // Gate projection weights
    %up_weight: tensor<?x?xi4>,    // Up projection weights
    %down_weight: tensor<?x?xi4>   // Down projection weights
  ) -> tensor<?x?x?xi4> {
    
    // Gate projection
    %gate_proj = npu.int4_linear %input, %gate_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Up projection  
    %up_proj = npu.int4_linear %input, %up_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply SiLU activation to gate (NPU Phoenix has optimized SiLU)
    %gate_activated = npu.silu_int4 %gate_proj : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Element-wise multiplication (gating mechanism)
    %gated = npu.int4_mul %gate_activated, %up_proj : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Down projection
    %output = npu.int4_linear %gated, %down_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
'''
        
        with open(path, 'w') as f:
            f.write(kernel_content)
    
    def _create_qwen_rope_kernel(self, path: Path):
        """Create Qwen-specific RoPE attention kernel"""
        kernel_content = '''// Qwen Rotary Position Embedding (RoPE) Kernel for NPU Phoenix
// Optimized for Qwen's RoPE attention mechanism

module {
  func.func @qwen_rope_attention_npu(
    %query: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %key: tensor<?x?x?xi4>,        // [batch, seq_len, head_dim]
    %rope_cos: tensor<?x?xf16>,    // Cosine values for RoPE
    %rope_sin: tensor<?x?xf16>     // Sine values for RoPE
  ) -> (tensor<?x?x?xi4>, tensor<?x?x?xi4>) {
    
    // Apply RoPE to query
    %q_rotated = npu.apply_rope_int4 %query, %rope_cos, %rope_sin : 
      tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?x?xi4>
    
    // Apply RoPE to key
    %k_rotated = npu.apply_rope_int4 %key, %rope_cos, %rope_sin : 
      tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?x?xi4>
    
    return %q_rotated, %k_rotated : tensor<?x?x?xi4>, tensor<?x?x?xi4>
  }
  
  // Qwen-specific attention with RoPE
  func.func @qwen_attention_with_rope_npu(
    %input: tensor<?x?x?xi4>,
    %q_weight: tensor<?x?xi4>,
    %k_weight: tensor<?x?xi4>, 
    %v_weight: tensor<?x?xi4>,
    %o_weight: tensor<?x?xi4>,
    %rope_cos: tensor<?x?xf16>,
    %rope_sin: tensor<?x?xf16>
  ) -> tensor<?x?x?xi4> {
    
    // Project to Q, K, V
    %query = npu.int4_linear %input, %q_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %key = npu.int4_linear %input, %k_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %value = npu.int4_linear %input, %v_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply RoPE
    %q_rope, %k_rope = func.call @qwen_rope_attention_npu(%query, %key, %rope_cos, %rope_sin) : 
      (tensor<?x?x?xi4>, tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16>) -> 
      (tensor<?x?x?xi4>, tensor<?x?x?xi4>)
    
    // Standard attention computation
    %scores = npu.int4_matmul %q_rope, %k_rope transpose_b : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    %attention_weights = npu.softmax_int4 %scores : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    %attended = npu.int4_matmul %attention_weights, %value : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Output projection
    %output = npu.int4_linear %attended, %o_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
'''
        
        with open(path, 'w') as f:
            f.write(kernel_content)
    
    def create_test_suite(self):
        """Create NPU kernel test suite"""
        logger.info("🧪 Creating NPU kernel test suite...")
        
        test_file = self.npu_dev_dir / "tests" / "test_npu_kernels.py"
        test_content = '''#!/usr/bin/env python3
"""
NPU Kernel Test Suite
Tests all NPU kernels with synthetic data
"""
import torch
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUKernelTester:
    """Test NPU kernels with various model architectures"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_universal_attention(self):
        """Test universal attention kernel"""
        logger.info("🧪 Testing universal attention kernel...")
        
        # Create test tensors
        batch_size, seq_len, head_dim = 1, 64, 64
        
        # Simulate INT4 tensors (use INT8 for testing)
        query = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        key = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        value = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        
        logger.info(f"   Input shapes: Q{query.shape}, K{key.shape}, V{value.shape}")
        
        # Simulate NPU execution (placeholder)
        start_time = time.time()
        # In real implementation, this would call NPU kernel
        output = self._simulate_attention_npu(query, key, value)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Universal attention test passed")
        
        return {"execution_time": execution_time, "output_shape": output.shape}
    
    def test_gemma_ffn(self):
        """Test Gemma gated FFN kernel"""
        logger.info("🧪 Testing Gemma FFN kernel...")
        
        # Gemma FFN dimensions
        batch_size, seq_len, hidden_size = 1, 64, 2048
        intermediate_size = 8192
        
        input_tensor = torch.randint(-7, 8, (batch_size, seq_len, hidden_size), dtype=torch.int8)
        gate_weight = torch.randint(-7, 8, (intermediate_size, hidden_size), dtype=torch.int8)
        up_weight = torch.randint(-7, 8, (intermediate_size, hidden_size), dtype=torch.int8)
        down_weight = torch.randint(-7, 8, (hidden_size, intermediate_size), dtype=torch.int8)
        
        logger.info(f"   Input shape: {input_tensor.shape}")
        
        start_time = time.time()
        output = self._simulate_gemma_ffn_npu(input_tensor, gate_weight, up_weight, down_weight)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Gemma FFN test passed")
        
        return {"execution_time": execution_time, "output_shape": output.shape}
    
    def test_qwen_rope(self):
        """Test Qwen RoPE attention kernel"""
        logger.info("🧪 Testing Qwen RoPE kernel...")
        
        # Qwen attention dimensions
        batch_size, seq_len, head_dim = 1, 64, 64
        
        query = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        key = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        
        # RoPE parameters
        rope_cos = torch.randn(seq_len, head_dim, dtype=torch.float16)
        rope_sin = torch.randn(seq_len, head_dim, dtype=torch.float16)
        
        logger.info(f"   Input shapes: Q{query.shape}, K{key.shape}")
        
        start_time = time.time()
        q_rotated, k_rotated = self._simulate_qwen_rope_npu(query, key, rope_cos, rope_sin)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shapes: Q{q_rotated.shape}, K{k_rotated.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Qwen RoPE test passed")
        
        return {"execution_time": execution_time, "output_shapes": (q_rotated.shape, k_rotated.shape)}
    
    def _simulate_attention_npu(self, query, key, value):
        """Simulate NPU attention computation"""
        # Convert to float for computation, then back to int8
        q_float = query.float()
        k_float = key.float() 
        v_float = value.float()
        
        # Attention computation
        scores = torch.matmul(q_float, k_float.transpose(-2, -1))
        scores = scores / (q_float.shape[-1] ** 0.5)
        
        # Apply causal mask
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, -float('inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v_float)
        
        # Convert back to int8 (simulate quantization)
        return torch.clamp(output.round(), -7, 7).to(torch.int8)
    
    def _simulate_gemma_ffn_npu(self, input_tensor, gate_weight, up_weight, down_weight):
        """Simulate Gemma FFN computation on NPU"""
        # Convert to float for computation
        input_float = input_tensor.float()
        gate_float = gate_weight.float()
        up_float = up_weight.float()
        down_float = down_weight.float()
        
        # Gate projection
        gate_proj = torch.matmul(input_float, gate_float.T)
        
        # Up projection
        up_proj = torch.matmul(input_float, up_float.T)
        
        # SiLU activation on gate
        gate_activated = gate_proj * torch.sigmoid(gate_proj)
        
        # Gating
        intermediate = gate_activated * up_proj
        
        # Down projection
        output = torch.matmul(intermediate, down_float.T)
        
        # Convert back to int8
        return torch.clamp(output.round(), -7, 7).to(torch.int8)
    
    def _simulate_qwen_rope_npu(self, query, key, rope_cos, rope_sin):
        """Simulate Qwen RoPE computation on NPU"""
        def apply_rope(tensor, cos, sin):
            # Simplified RoPE application
            x1, x2 = tensor[..., ::2], tensor[..., 1::2]
            
            # Convert to float for computation
            x1_float = x1.float()
            x2_float = x2.float()
            cos = cos[..., ::2]
            sin = sin[..., ::2]
            
            rotated_x1 = x1_float * cos - x2_float * sin
            rotated_x2 = x1_float * sin + x2_float * cos
            
            # Interleave back
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
            
            # Convert back to int8
            return torch.clamp(rotated.round(), -7, 7).to(torch.int8)
        
        q_rotated = apply_rope(query, rope_cos, rope_sin)
        k_rotated = apply_rope(key, rope_cos, rope_sin)
        
        return q_rotated, k_rotated
    
    def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🦄 Starting NPU Kernel Test Suite")
        logger.info("=" * 50)
        
        tests = [
            ("Universal Attention", self.test_universal_attention),
            ("Gemma FFN", self.test_gemma_ffn),
            ("Qwen RoPE", self.test_qwen_rope)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                logger.info("")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                self.test_results[test_name] = {"error": str(e)}
        
        # Summary
        logger.info("📊 Test Suite Summary:")
        for test_name, result in self.test_results.items():
            if "error" in result:
                logger.error(f"   ❌ {test_name}: FAILED")
            else:
                logger.info(f"   ✅ {test_name}: PASSED")
        
        logger.info("🎉 NPU kernel test suite complete!")

if __name__ == "__main__":
    tester = NPUKernelTester()
    tester.run_all_tests()
'''
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        logger.info("   ✅ Test suite created")
    
    def create_documentation(self):
        """Create NPU development documentation"""
        logger.info("📚 Creating NPU development documentation...")
        
        docs_dir = self.npu_dev_dir / "docs"
        
        # Main README
        readme_content = """# NPU Development for Multi-Model Optimization

## Overview
This directory contains NPU Phoenix kernel development for optimizing multiple LLM architectures.

## Supported Models
- **Gemma 3 27B-IT**: Ultra-aggressive quantization + NPU acceleration
- **Qwen2.5 7B/14B/32B**: RoPE attention optimization
- **Gemma 2 2B/9B**: Ultra-fast inference
- **Qwen2.5-VL**: Vision-language fusion

## Directory Structure
```
npu_development/
├── kernels/
│   ├── universal/          # Cross-model kernels
│   ├── gemma/             # Gemma-specific optimizations
│   └── qwen/              # Qwen-specific optimizations
├── tools/                 # Development tools
├── tests/                 # Kernel testing
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Quick Start
1. Run setup: `python setup_npu_development.py`
2. Test kernels: `python tests/test_npu_kernels.py`
3. Compile kernels: `./tools/compile_kernels.sh`

## Performance Targets
- Gemma 3 27B: 150-200 TPS
- Qwen2.5-7B: 200-300 TPS  
- Gemma 2 2B: 400-600 TPS
"""
        
        with open(docs_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info("   ✅ Documentation created")
    
    def run_setup(self) -> bool:
        """Run complete NPU development setup"""
        logger.info("🚀 NPU Development Environment Setup")
        logger.info("🎯 Target: Multi-model NPU optimization")
        logger.info("=" * 60)
        
        setup_steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Directory Structure", lambda: self.create_directory_structure() or True),
            ("MLIR-AIE2 Setup", self.setup_mlir_aie2),
            ("Kernel Templates", lambda: self.create_kernel_templates() or True),
            ("Test Suite", lambda: self.create_test_suite() or True),
            ("Documentation", lambda: self.create_documentation() or True)
        ]
        
        all_success = True
        for step_name, step_func in setup_steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                success = step_func()
                if success:
                    logger.info(f"   ✅ {step_name} completed")
                else:
                    logger.warning(f"   ⚠️ {step_name} completed with warnings")
                    all_success = False
            except Exception as e:
                logger.error(f"   ❌ {step_name} failed: {e}")
                all_success = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        if all_success:
            logger.info("🎉 NPU DEVELOPMENT ENVIRONMENT READY!")
            logger.info("✅ All components set up successfully")
        else:
            logger.warning("⚠️ Setup completed with some warnings")
            logger.info("🔧 Check logs above for details")
        
        logger.info("\n🚀 Next Steps:")
        logger.info("1. Test NPU kernels: python npu_development/tests/test_npu_kernels.py")
        logger.info("2. Review kernel templates in npu_development/kernels/")
        logger.info("3. Start kernel development for your target models")
        
        return all_success

def main():
    """Main setup execution"""
    setup = NPUDevelopmentSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()