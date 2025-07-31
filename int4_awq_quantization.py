#!/usr/bin/env python3
"""
INT4 AWQ (Activation-aware Weight Quantization) Implementation
For Magic Unicorn System - Maximum model compression with minimal accuracy loss
Based on Gemini's research findings
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Types of quantization schemes"""
    INT4_AWQ = "int4_awq"
    INT4_GPTQ = "int4_gptq"
    INT8_DYNAMIC = "int8_dynamic"
    INT2_EXTREME = "int2_extreme"

@dataclass
class QuantizationConfig:
    """Configuration for AWQ quantization"""
    quantization_type: QuantizationType
    bits: int = 4
    group_size: int = 128
    clip_ratio: float = 1.0
    zero_point: bool = True
    symmetric: bool = False
    activation_aware: bool = True
    preserve_accuracy_layers: List[str] = None

@dataclass
class QuantizedLayer:
    """Quantized layer representation"""
    name: str
    original_weight: torch.Tensor
    quantized_weight: torch.Tensor
    scales: torch.Tensor
    zeros: Optional[torch.Tensor]
    bits: int
    group_size: int
    compression_ratio: float

class INT4AWQQuantizer:
    """
    🦄 Magic Unicorn INT4 AWQ Quantizer
    
    Features:
    - Activation-aware weight quantization for minimal accuracy loss
    - INT4 quantization with groupwise scaling
    - Automatic calibration dataset selection
    - Hardware-optimized kernels for NPU/GPU
    - Maximum compression with quality preservation
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[QuantizationConfig] = None,
                 calibration_samples: int = 128):
        """
        Initialize AWQ quantizer
        
        Args:
            model_path: Path to model to quantize
            config: Quantization configuration
            calibration_samples: Number of samples for calibration
        """
        
        self.model_path = model_path
        self.config = config or self._get_default_config()
        self.calibration_samples = calibration_samples
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        # Quantization state
        self.calibration_data = []
        self.activation_scales = {}
        self.quantized_layers = {}
        
        # Performance tracking
        self.original_model_size = 0
        self.quantized_model_size = 0
        self.quantization_time = 0.0
        
        logger.info("🦄 INT4 AWQ Quantizer initializing...")
        
    def _get_default_config(self) -> QuantizationConfig:
        """Get default AWQ quantization configuration"""
        
        return QuantizationConfig(
            quantization_type=QuantizationType.INT4_AWQ,
            bits=4,
            group_size=128,
            clip_ratio=1.0,
            zero_point=True,
            symmetric=False,
            activation_aware=True,
            preserve_accuracy_layers=[
                "embed_tokens",    # Embedding layers critical for accuracy
                "lm_head",         # Output projection critical
                "norm"             # Layer norms typically not quantized
            ]
        )
    
    def load_model(self) -> bool:
        """Load model for quantization"""
        
        try:
            # Import LZMA fallback to handle missing _lzma module
            import sys
            sys.path.insert(0, '/home/ucadmin/Development/Unicorn-Execution-Engine')
            from lzma_fallback import ensure_lzma_available
            ensure_lzma_available()
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
            from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
            
            logger.info(f"📦 Loading model from {self.model_path}")
            
            # Check if model path exists and contains config.json
            if not Path(self.model_path).exists() or not (Path(self.model_path) / "config.json").exists():
                logger.error(f"Model path does not exist or is not a valid Hugging Face model directory: {self.model_path}")
                return False

            try:
                # Try loading with AutoModelForCausalLM first
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu",  # Keep on CPU for quantization
                    trust_remote_code=True,
                    use_safetensors=True  # Prefer safetensors to avoid pickle/lzma issues
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=False  # Avoid potential tokenizer compression issues
                )
            except Exception as e_auto:
                logger.warning(f"AutoModelForCausalLM failed ({e_auto}), trying direct GemmaForCausalLM import.")
                try:
                    # Fallback to direct GemmaForCausalLM import
                    self.model = GemmaForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="cpu",
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    self.tokenizer = GemmaTokenizer.from_pretrained(
                        self.model_path,
                        use_fast=False
                    )
                except Exception as e_gemma:
                    logger.error(f"Direct GemmaForCausalLM import also failed: {e_gemma}")
                    return False
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Calculate original model size
            self.original_model_size = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            
            logger.info(f"✅ Model loaded: {self.original_model_size / 1024**3:.2f}GB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def prepare_calibration_data(self) -> List[torch.Tensor]:
        """Prepare calibration dataset for activation-aware quantization"""
        
        try:
            logger.info(f"📊 Preparing calibration data ({self.calibration_samples} samples)...")
            
            # Sample calibration prompts (diverse set for good coverage)
            calibration_prompts = [
                "What is the capital of France?",
                "Explain quantum physics in simple terms.",
                "Write a short story about a magical forest.",
                "How do neural networks learn?",
                "Describe the process of photosynthesis.",
                "What are the benefits of renewable energy?",
                "Explain the theory of relativity.",
                "Write a poem about the ocean.",
                "How does machine learning work?",
                "Describe the human brain.",
                "What is artificial intelligence?",
                "Explain climate change and its effects.",
                "Write a recipe for chocolate cake.",
                "How do computers process information?",
                "Describe the solar system."
            ]
            
            # Extend prompts to reach desired sample count
            extended_prompts = (calibration_prompts * (self.calibration_samples // len(calibration_prompts) + 1))[:self.calibration_samples]
            
            calibration_data = []
            
            for i, prompt in enumerate(extended_prompts):
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                calibration_data.append(inputs["input_ids"])
                
                if (i + 1) % 32 == 0:
                    logger.debug(f"   Prepared {i + 1}/{len(extended_prompts)} calibration samples")
            
            self.calibration_data = calibration_data
            logger.info(f"✅ Calibration data ready: {len(calibration_data)} samples")
            return calibration_data
            
        except Exception as e:
            logger.error(f"❌ Calibration data preparation failed: {e}")
            return []
    
    def collect_activation_statistics(self) -> Dict[str, torch.Tensor]:
        """Collect activation statistics for AWQ quantization"""
        
        logger.info("📈 Collecting activation statistics...")
        
        activation_scales = {}
        
        # Hook to collect activations
        def activation_hook(name):
            def hook(module, input, output):
                if name not in activation_scales:
                    activation_scales[name] = []
                
                # Calculate activation scales (for AWQ)
                if isinstance(output, torch.Tensor):
                    # Use percentile-based scaling for better outlier handling
                    scale = torch.quantile(torch.abs(output.flatten()), 0.99)
                    activation_scales[name].append(scale.cpu())
                elif isinstance(output, tuple):
                    # Handle tuple outputs (like attention)
                    scale = torch.quantile(torch.abs(output[0].flatten()), 0.99)
                    activation_scales[name].append(scale.cpu())
            
            return hook
        
        # Register hooks for all quantizable layers
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if not self._should_preserve_layer(name):
                    hook = module.register_forward_hook(activation_hook(name))
                    hooks.append(hook)
        
        # Run calibration samples through model
        self.model.eval()
        with torch.no_grad():
            for i, input_ids in enumerate(self.calibration_data):
                try:
                    _ = self.model(input_ids)
                    
                    if (i + 1) % 32 == 0:
                        logger.debug(f"   Processed {i + 1}/{len(self.calibration_data)} calibration samples")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Calibration sample {i} failed: {e}")
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate final scales (median of collected scales)
        final_scales = {}
        for name, scales in activation_scales.items():
            if scales:
                final_scales[name] = torch.median(torch.stack(scales))
            else:
                final_scales[name] = torch.tensor(1.0)
        
        self.activation_scales = final_scales
        logger.info(f"✅ Activation statistics collected for {len(final_scales)} layers")
        
        return final_scales
    
    def quantize_layer_awq(self, 
                          name: str, 
                          layer: nn.Module, 
                          activation_scale: torch.Tensor) -> Optional[QuantizedLayer]:
        """
        Quantize a single layer using AWQ method
        
        Args:
            name: Layer name
            layer: Layer module
            activation_scale: Activation scale for this layer
            
        Returns:
            QuantizedLayer object or None if not quantizable
        """
        
        if not isinstance(layer, nn.Linear):
            return None
        
        if self._should_preserve_layer(name):
            logger.debug(f"🔒 Preserving layer: {name}")
            return None
        
        try:
            weight = layer.weight.data.clone()
            
            # AWQ: Scale weights by activation scale to minimize quantization error
            # This is the key insight of AWQ - weight importance varies by activation magnitude
            awq_weight = weight * activation_scale.unsqueeze(0)
            
            # Group-wise quantization
            group_size = self.config.group_size
            bits = self.config.bits
            
            # Reshape for group-wise processing
            original_shape = awq_weight.shape
            if awq_weight.numel() % group_size != 0:
                # Pad to make divisible by group_size
                pad_size = group_size - (awq_weight.numel() % group_size)
                awq_weight = torch.cat([awq_weight.flatten(), torch.zeros(pad_size)])
            else:
                awq_weight = awq_weight.flatten()
            
            awq_weight = awq_weight.reshape(-1, group_size)
            
            # Calculate scales and zero points per group
            if self.config.symmetric:
                # Symmetric quantization
                scales = torch.max(torch.abs(awq_weight), dim=1)[0] / (2**(bits-1) - 1)
                zeros = None
            else:
                # Asymmetric quantization
                min_vals = torch.min(awq_weight, dim=1)[0]
                max_vals = torch.max(awq_weight, dim=1)[0]
                
                scales = (max_vals - min_vals) / (2**bits - 1)
                zeros = -min_vals / scales
                zeros = torch.round(zeros).clamp(0, 2**bits - 1)
            
            # Quantize weights
            if zeros is not None:
                quantized = torch.round(awq_weight / scales.unsqueeze(1) + zeros.unsqueeze(1))
            else:
                quantized = torch.round(awq_weight / scales.unsqueeze(1))
            
            # Clamp to valid range
            quantized = quantized.clamp(0, 2**bits - 1)
            
            # Pack bits for storage efficiency
            if bits == 4:
                # Pack two 4-bit values into one uint8
                quantized_packed = self._pack_4bit(quantized)
            else:
                quantized_packed = quantized.to(torch.uint8)
            
            # Reshape back to original (accounting for padding)
            if quantized.numel() > np.prod(original_shape):
                quantized = quantized.flatten()[:np.prod(original_shape)]
                quantized_packed = quantized_packed.flatten()[:np.prod(original_shape)]
            
            quantized_packed = quantized_packed.reshape(original_shape)
            
            # Calculate compression ratio
            original_size = weight.numel() * weight.element_size()
            quantized_size = (
                quantized_packed.numel() * quantized_packed.element_size() +
                scales.numel() * scales.element_size() +
                (zeros.numel() * zeros.element_size() if zeros is not None else 0)
            )
            compression_ratio = original_size / quantized_size
            
            quantized_layer = QuantizedLayer(
                name=name,
                original_weight=weight,
                quantized_weight=quantized_packed,
                scales=scales,
                zeros=zeros,
                bits=bits,
                group_size=group_size,
                compression_ratio=compression_ratio
            )
            
            logger.debug(f"⚡ Quantized {name}: {compression_ratio:.1f}x compression")
            return quantized_layer
            
        except Exception as e:
            logger.error(f"❌ Layer quantization failed for {name}: {e}")
            return None
    
    def _pack_4bit(self, quantized: torch.Tensor) -> torch.Tensor:
        """Pack 4-bit values into uint8 for storage efficiency"""
        
        # Ensure even number of elements
        if quantized.numel() % 2 != 0:
            quantized = torch.cat([quantized.flatten(), torch.zeros(1)])
        
        quantized = quantized.flatten().to(torch.uint8)
        
        # Pack pairs of 4-bit values
        packed = torch.zeros(quantized.numel() // 2, dtype=torch.uint8)
        for i in range(0, quantized.numel(), 2):
            packed[i // 2] = (quantized[i] & 0xF) | ((quantized[i + 1] & 0xF) << 4)
        
        return packed
    
    def _should_preserve_layer(self, layer_name: str) -> bool:
        """Check if layer should be preserved (not quantized)"""
        
        if self.config.preserve_accuracy_layers:
            for preserve_pattern in self.config.preserve_accuracy_layers:
                if preserve_pattern in layer_name:
                    return True
        
        return False
    
    def quantize_model(self) -> bool:
        """Quantize entire model using AWQ"""
        
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting AWQ quantization...")
            
            # Prepare calibration data
            if not self.prepare_calibration_data():
                return False
            
            # Collect activation statistics
            activation_scales = self.collect_activation_statistics()
            
            if not activation_scales:
                logger.error("❌ No activation scales collected")
                return False
            
            # Quantize each layer
            quantized_count = 0
            preserved_count = 0
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    
                    activation_scale = activation_scales.get(name, torch.tensor(1.0))
                    
                    quantized_layer = self.quantize_layer_awq(name, module, activation_scale)
                    
                    if quantized_layer:
                        self.quantized_layers[name] = quantized_layer
                        quantized_count += 1
                    else:
                        preserved_count += 1
            
            # Calculate final model size
            self.quantized_model_size = self._calculate_quantized_size()
            self.quantization_time = time.time() - start_time
            
            compression_ratio = self.original_model_size / self.quantized_model_size
            
            logger.info(f"✅ AWQ Quantization complete!")
            logger.info(f"📊 Quantized {quantized_count} layers, preserved {preserved_count} layers")
            logger.info(f"💾 Model size: {self.original_model_size / 1024**3:.2f}GB → {self.quantized_model_size / 1024**3:.2f}GB")
            logger.info(f"🎯 Compression ratio: {compression_ratio:.1f}x")
            logger.info(f"⏱️  Quantization time: {self.quantization_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model quantization failed: {e}")
            return False
    
    def _calculate_quantized_size(self) -> int:
        """Calculate total size of quantized model"""
        
        quantized_size = 0
        
        # Size of quantized layers
        for layer in self.quantized_layers.values():
            quantized_size += (
                layer.quantized_weight.numel() * layer.quantized_weight.element_size() +
                layer.scales.numel() * layer.scales.element_size() +
                (layer.zeros.numel() * layer.zeros.element_size() if layer.zeros is not None else 0)
            )
        
        # Size of non-quantized parameters
        for name, param in self.model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])  # Remove parameter name
            if layer_name not in self.quantized_layers:
                quantized_size += param.numel() * param.element_size()
        
        return quantized_size
    
    def save_quantized_model(self, output_path: str) -> bool:
        """Save quantized model to disk"""
        
        try:
            logger.info(f"💾 Saving quantized model to {output_path}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save quantized layers
            quantized_data = {}
            for name, layer in self.quantized_layers.items():
                quantized_data[name] = {
                    'quantized_weight': layer.quantized_weight,
                    'scales': layer.scales,
                    'zeros': layer.zeros,
                    'bits': layer.bits,
                    'group_size': layer.group_size,
                    'compression_ratio': layer.compression_ratio
                }
            
            torch.save(quantized_data, os.path.join(output_path, "quantized_layers.pt"))
            
            # Save quantization config
            config_dict = {
                'quantization_type': self.config.quantization_type.value,
                'bits': self.config.bits,
                'group_size': self.config.group_size,
                'clip_ratio': self.config.clip_ratio,
                'zero_point': self.config.zero_point,
                'symmetric': self.config.symmetric,
                'activation_aware': self.config.activation_aware,
                'preserve_accuracy_layers': self.config.preserve_accuracy_layers,
                'original_model_size': self.original_model_size,
                'quantized_model_size': self.quantized_model_size,
                'compression_ratio': self.original_model_size / self.quantized_model_size,
                'quantization_time': self.quantization_time
            }
            
            with open(os.path.join(output_path, "quantization_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Save original model config (for reconstruction)
            self.model.config.save_pretrained(output_path)
            
            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"✅ Quantized model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
            return False
    
    def create_inference_optimized_kernels(self, output_path: str) -> bool:
        """Create optimized kernels for INT4 inference on NPU/GPU"""
        
        try:
            logger.info("⚡ Creating optimized INT4 inference kernels...")
            
            # Create kernel source for INT4 AWQ inference
            vulkan_kernel = self._create_vulkan_int4_kernel()
            npu_kernel = self._create_npu_int4_kernel()
            
            kernels_dir = os.path.join(output_path, "kernels")
            os.makedirs(kernels_dir, exist_ok=True)
            
            # Save Vulkan kernel for GPU
            with open(os.path.join(kernels_dir, "int4_awq_gpu.comp"), 'w') as f:
                f.write(vulkan_kernel)
            
            # Save NPU kernel template
            with open(os.path.join(kernels_dir, "int4_awq_npu.mlir"), 'w') as f:
                f.write(npu_kernel)
            
            logger.info("✅ Optimized kernels created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Kernel creation failed: {e}")
            return False
    
    def _create_vulkan_int4_kernel(self) -> str:
        """Create Vulkan compute shader for INT4 AWQ inference"""
        
        return '''#version 450

// INT4 AWQ Optimized Matrix Multiplication Kernel
// Handles packed 4-bit weights with group-wise scaling

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputBuffer {
    float input_data[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer WeightBuffer {
    uint packed_weights[];  // 4-bit weights packed into uint8
};

layout(set = 0, binding = 2, std430) restrict readonly buffer ScaleBuffer {
    float scales[];
};

layout(set = 0, binding = 3, std430) restrict readonly buffer ZeroBuffer {
    float zeros[];
};

layout(set = 0, binding = 4, std430) restrict writeonly buffer OutputBuffer {
    float output_data[];
};

layout(push_constant) uniform PushConstants {
    uint M;  // Input rows
    uint N;  // Output columns  
    uint K;  // Input columns
    uint group_size;
};

shared float input_cache[16][16];
shared float weight_cache[16][16];

// Unpack 4-bit weight from packed uint8
float unpack_weight(uint packed_value, uint index) {
    uint shift = (index & 1) * 4;
    uint weight_4bit = (packed_value >> shift) & 0xF;
    return float(weight_4bit);
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (row >= M || col >= N) return;
    
    float accumulator = 0.0;
    
    // Process in tiles for cache efficiency
    for (uint tile = 0; tile < K; tile += 16) {
        
        // Load input tile to shared memory
        uint input_row = gl_LocalInvocationID.y;
        uint input_col = gl_LocalInvocationID.x;
        
        if (tile + input_col < K) {
            input_cache[input_row][input_col] = input_data[row * K + tile + input_col];
        } else {
            input_cache[input_row][input_col] = 0.0;
        }
        
        // Load and unpack weight tile to shared memory
        uint weight_row = gl_LocalInvocationID.y;
        uint weight_col = gl_LocalInvocationID.x;
        
        if (tile + weight_row < K) {
            uint packed_idx = ((tile + weight_row) * N + col) / 2;
            uint packed_value = packed_weights[packed_idx];
            uint weight_idx = ((tile + weight_row) * N + col) % 2;
            
            float quantized_weight = unpack_weight(packed_value, weight_idx);
            
            // Apply AWQ scaling and zero point
            uint group_idx = (tile + weight_row) / group_size;
            float scale = scales[group_idx * N + col];
            float zero = zeros[group_idx * N + col];
            
            weight_cache[weight_row][weight_col] = (quantized_weight - zero) * scale;
        } else {
            weight_cache[weight_row][weight_col] = 0.0;
        }
        
        barrier();
        
        // Compute partial dot product
        for (uint k = 0; k < 16; ++k) {
            accumulator += input_cache[gl_LocalInvocationID.y][k] * 
                          weight_cache[k][gl_LocalInvocationID.x];
        }
        
        barrier();
    }
    
    output_data[row * N + col] = accumulator;
}
'''
    
    def _create_npu_int4_kernel(self) -> str:
        """Create NPU MLIR kernel for INT4 AWQ inference"""
        
        return '''// INT4 AWQ NPU Kernel for AMD XDNA Architecture
// Optimized for 4-bit weight computation with activation awareness

module {
  func.func @int4_awq_matmul(
    %input: memref<?x?xf32>,
    %packed_weights: memref<?x?xi8>,
    %scales: memref<?x?xf32>,
    %zeros: memref<?x?xf32>,
    %output: memref<?x?xf32>
  ) {
    
    // Get dimensions
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    
    %M = memref.dim %input, %c0 : memref<?x?xf32>
    %K = memref.dim %input, %c1 : memref<?x?xf32>
    %N = memref.dim %output, %c1 : memref<?x?xf32>
    
    // Main computation loops
    scf.parallel (%i, %j) = (%c0, %c0) to (%M, %N) step (%c1, %c1) {
      
      %sum = arith.constant 0.0 : f32
      
      // Inner loop over K dimension with INT4 unpacking
      %final_sum = scf.for %k = %c0 to %K step %c1 iter_args(%acc = %sum) -> (f32) {
        
        // Load input value
        %input_val = memref.load %input[%i, %k] : memref<?x?xf32>
        
        // Unpack 4-bit weight
        %packed_idx = arith.divui %k, %c2 : index
        %weight_offset = arith.remui %k, %c2 : index
        %packed_weight = memref.load %packed_weights[%j, %packed_idx] : memref<?x?xi8>
        
        // Extract 4-bit value
        %shift_amount = arith.muli %weight_offset, %c4 : index
        %shift_amount_i8 = arith.index_cast %shift_amount : index to i8
        %shifted = arith.shli %packed_weight, %shift_amount_i8 : i8
        %mask = arith.constant 15 : i8  // 0xF
        %weight_4bit = arith.andi %shifted, %mask : i8
        %weight_float = arith.sitofp %weight_4bit : i8 to f32
        
        // Apply AWQ scaling
        %group_size = arith.constant 128 : index
        %group_idx = arith.divui %k, %group_size : index
        %scale = memref.load %scales[%group_idx, %j] : memref<?x?xf32>
        %zero = memref.load %zeros[%group_idx, %j] : memref<?x?xf32>
        
        // Dequantize: (quantized - zero) * scale
        %dequant_temp = arith.subf %weight_float, %zero : f32
        %dequant_weight = arith.mulf %dequant_temp, %scale : f32
        
        // Accumulate
        %product = arith.mulf %input_val, %dequant_weight : f32
        %new_acc = arith.addf %acc, %product : f32
        
        scf.yield %new_acc : f32
      }
      
      memref.store %final_sum, %output[%i, %j] : memref<?x?xf32>
    }
    
    return
  }
}
'''

def quantize_gemma3_4b_int4():
    """Quantize Gemma3 4B model to INT4 AWQ"""
    
    logger.info("🚀 Starting Gemma3 4B INT4 AWQ Quantization...")
    
    # Initialize quantizer
    quantizer = INT4AWQQuantizer(
        model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized",
        calibration_samples=256
    )
    
    # Load model
    if not quantizer.load_model():
        logger.error("❌ Failed to load model")
        return False
    
    # Quantize model
    if not quantizer.quantize_model():
        logger.error("❌ Failed to quantize model")
        return False
    
    # Save quantized model
    output_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-int4-awq"
    if not quantizer.save_quantized_model(output_path):
        logger.error("❌ Failed to save quantized model")
        return False
    
    # Create optimized kernels
    if not quantizer.create_inference_optimized_kernels(output_path):
        logger.warning("⚠️  Failed to create optimized kernels")
    
    logger.info("✅ Gemma3 4B INT4 AWQ quantization complete!")
    return True

if __name__ == "__main__":
    quantize_gemma3_4b_int4()