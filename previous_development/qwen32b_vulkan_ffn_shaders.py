#!/usr/bin/env python3
"""
Qwen 2.5 32B Vulkan FFN Compute Shaders
Optimized SPIR-V shaders for AMD Radeon 780M iGPU
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen32BVulkanFFNShaders:
    """Vulkan compute shaders for Qwen 2.5 32B FFN layers"""
    
    def __init__(self):
        self.gpu_config = {
            "device": "AMD Radeon 780M",
            "compute_units": 12,
            "memory": 16 * 1024**3,  # 16GB DDR5
            "precision": "INT4",
            "vulkan_version": "1.3"
        }
        
        self.qwen32b_ffn_config = {
            "hidden_size": 5120,
            "intermediate_size": 27392,
            "num_layers": 64,
            "activation": "silu",  # SiLU activation function
            "precision": "INT4_GROUPED"
        }
        
        self.shader_templates = self.create_shader_templates()
        
    def create_shader_templates(self) -> Dict:
        """Create GLSL compute shader templates for FFN operations"""
        
        templates = {
            "gate_projection": self.create_gate_projection_shader(),
            "up_projection": self.create_up_projection_shader(),
            "silu_activation": self.create_silu_activation_shader(),
            "down_projection": self.create_down_projection_shader(),
            "layer_norm": self.create_layer_norm_shader()
        }
        
        return templates
    
    def create_gate_projection_shader(self) -> str:
        """GLSL compute shader for gate projection with INT4 quantization"""
        
        shader = """#version 450

// Qwen 32B Gate Projection Shader for Radeon 780M
// INT4 grouped quantization optimized for iGPU

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input/Output buffers
layout(std430, binding = 0) readonly buffer InputBuffer {
    int input_data[];
};

layout(std430, binding = 1) readonly buffer WeightBuffer {
    uint weight_data[];  // Packed INT4 weights (2 weights per uint)
};

layout(std430, binding = 2) readonly buffer ScaleBuffer {
    float scale_data[];  // Quantization scales per group
};

layout(std430, binding = 3) readonly buffer ZeroPointBuffer {
    float zero_point_data[];  // Zero points per group
};

layout(std430, binding = 4) writeonly buffer OutputBuffer {
    int output_data[];
};

// Uniform parameters
layout(push_constant) uniform PushConstants {
    uint input_size;      // 5120
    uint output_size;     // 27392
    uint group_size;      // 128 (group size for quantization)
    uint num_groups;      // 5120 / 128 = 40
};

// Unpack INT4 weights from packed uint
uvec2 unpack_int4(uint packed_data, uint index) {
    uint shift = (index % 8) * 4;
    uint mask = 0xFu << shift;
    uint weight = (packed_data & mask) >> shift;
    
    // Convert 4-bit unsigned to signed
    if (weight > 7) {
        weight = weight - 16;
    }
    
    return uvec2(weight, (index + 1) % 8 == 0 ? 1 : 0);
}

void main() {
    uint output_idx = gl_GlobalInvocationID.y * gl_WorkGroupSize.x * gl_NumWorkGroups.x + gl_GlobalInvocationID.x;
    
    if (output_idx >= output_size) return;
    
    int accumulator = 0;
    
    // Compute dot product with grouped quantization
    for (uint input_idx = 0; input_idx < input_size; input_idx++) {
        // Determine group for this input
        uint group_idx = input_idx / group_size;
        
        // Get quantization parameters for this group
        float scale = scale_data[group_idx];
        float zero_point = zero_point_data[group_idx];
        
        // Get input value
        int input_val = input_data[input_idx];
        
        // Get packed weight index
        uint weight_matrix_idx = output_idx * input_size + input_idx;
        uint packed_idx = weight_matrix_idx / 2;
        uint weight_sub_idx = weight_matrix_idx % 2;
        
        // Unpack INT4 weight
        uint packed_weight = weight_data[packed_idx];
        uvec2 unpacked = unpack_int4(packed_weight, weight_sub_idx);
        int weight_val = int(unpacked.x);
        
        // Dequantize weight: weight_float = scale * (weight_int4 - zero_point)
        float weight_float = scale * (float(weight_val) - zero_point);
        
        // Accumulate (convert back to int for efficiency)
        accumulator += input_val * int(weight_float * 128.0);  // Scale up for precision
    }
    
    output_data[output_idx] = accumulator;
}
"""
        return shader
    
    def create_up_projection_shader(self) -> str:
        """GLSL compute shader for up projection"""
        
        shader = """#version 450

// Qwen 32B Up Projection Shader for Radeon 780M
// Similar to gate projection but separate weights

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer {
    int input_data[];
};

layout(std430, binding = 1) readonly buffer WeightBuffer {
    uint weight_data[];  // Packed INT4 weights
};

layout(std430, binding = 2) readonly buffer ScaleBuffer {
    float scale_data[];
};

layout(std430, binding = 3) readonly buffer ZeroPointBuffer {
    float zero_point_data[];
};

layout(std430, binding = 4) writeonly buffer OutputBuffer {
    int output_data[];
};

layout(push_constant) uniform PushConstants {
    uint input_size;      // 5120
    uint output_size;     // 27392
    uint group_size;      // 128
    uint num_groups;      // 40
};

uvec2 unpack_int4(uint packed_data, uint index) {
    uint shift = (index % 8) * 4;
    uint mask = 0xFu << shift;
    uint weight = (packed_data & mask) >> shift;
    
    if (weight > 7) {
        weight = weight - 16;
    }
    
    return uvec2(weight, (index + 1) % 8 == 0 ? 1 : 0);
}

void main() {
    uint output_idx = gl_GlobalInvocationID.y * gl_WorkGroupSize.x * gl_NumWorkGroups.x + gl_GlobalInvocationID.x;
    
    if (output_idx >= output_size) return;
    
    int accumulator = 0;
    
    for (uint input_idx = 0; input_idx < input_size; input_idx++) {
        uint group_idx = input_idx / group_size;
        float scale = scale_data[group_idx];
        float zero_point = zero_point_data[group_idx];
        
        int input_val = input_data[input_idx];
        
        uint weight_matrix_idx = output_idx * input_size + input_idx;
        uint packed_idx = weight_matrix_idx / 2;
        uint weight_sub_idx = weight_matrix_idx % 2;
        
        uint packed_weight = weight_data[packed_idx];
        uvec2 unpacked = unpack_int4(packed_weight, weight_sub_idx);
        int weight_val = int(unpacked.x);
        
        float weight_float = scale * (float(weight_val) - zero_point);
        accumulator += input_val * int(weight_float * 128.0);
    }
    
    output_data[output_idx] = accumulator;
}
"""
        return shader
    
    def create_silu_activation_shader(self) -> str:
        """GLSL compute shader for SiLU activation function"""
        
        shader = """#version 450

// Qwen 32B SiLU Activation Shader for Radeon 780M
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer GateBuffer {
    int gate_data[];
};

layout(std430, binding = 1) readonly buffer UpBuffer {
    int up_data[];
};

layout(std430, binding = 2) writeonly buffer OutputBuffer {
    int output_data[];
};

layout(push_constant) uniform PushConstants {
    uint data_size;       // 27392
    float scale_factor;   // Scale for integer arithmetic
};

// Fast SiLU approximation optimized for GPU
float fast_silu(float x) {
    // Clamp input to prevent overflow
    x = clamp(x, -10.0, 10.0);
    
    // Fast sigmoid approximation: sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
    float tanh_half_x = tanh(x * 0.5);
    float sigmoid_x = 0.5 + 0.5 * tanh_half_x;
    
    // SiLU = x * sigmoid(x)
    return x * sigmoid_x;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= data_size) return;
    
    // Convert INT to float for computation
    float gate_val = float(gate_data[idx]) / scale_factor;
    float up_val = float(up_data[idx]) / scale_factor;
    
    // Apply SiLU to gate values
    float silu_gate = fast_silu(gate_val);
    
    // Element-wise multiplication: silu(gate) * up
    float result = silu_gate * up_val;
    
    // Convert back to INT
    output_data[idx] = int(result * scale_factor);
}
"""
        return shader
    
    def create_down_projection_shader(self) -> str:
        """GLSL compute shader for down projection"""
        
        shader = """#version 450

// Qwen 32B Down Projection Shader for Radeon 780M
// Project from intermediate size (27392) back to hidden size (5120)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer {
    int input_data[];
};

layout(std430, binding = 1) readonly buffer WeightBuffer {
    uint weight_data[];  // Packed INT4 weights
};

layout(std430, binding = 2) readonly buffer ScaleBuffer {
    float scale_data[];
};

layout(std430, binding = 3) readonly buffer ZeroPointBuffer {
    float zero_point_data[];
};

layout(std430, binding = 4) writeonly buffer OutputBuffer {
    int output_data[];
};

layout(push_constant) uniform PushConstants {
    uint input_size;      // 27392
    uint output_size;     // 5120
    uint group_size;      // 128
    uint num_groups;      // 214 (27392 / 128)
};

uvec2 unpack_int4(uint packed_data, uint index) {
    uint shift = (index % 8) * 4;
    uint mask = 0xFu << shift;
    uint weight = (packed_data & mask) >> shift;
    
    if (weight > 7) {
        weight = weight - 16;
    }
    
    return uvec2(weight, (index + 1) % 8 == 0 ? 1 : 0);
}

void main() {
    uint output_idx = gl_GlobalInvocationID.y * gl_WorkGroupSize.x * gl_NumWorkGroups.x + gl_GlobalInvocationID.x;
    
    if (output_idx >= output_size) return;
    
    int accumulator = 0;
    
    for (uint input_idx = 0; input_idx < input_size; input_idx++) {
        uint group_idx = input_idx / group_size;
        float scale = scale_data[group_idx];
        float zero_point = zero_point_data[group_idx];
        
        int input_val = input_data[input_idx];
        
        uint weight_matrix_idx = output_idx * input_size + input_idx;
        uint packed_idx = weight_matrix_idx / 2;
        uint weight_sub_idx = weight_matrix_idx % 2;
        
        uint packed_weight = weight_data[packed_idx];
        uvec2 unpacked = unpack_int4(packed_weight, weight_sub_idx);
        int weight_val = int(unpacked.x);
        
        float weight_float = scale * (float(weight_val) - zero_point);
        accumulator += input_val * int(weight_float * 128.0);
    }
    
    output_data[output_idx] = accumulator;
}
"""
        return shader
    
    def create_layer_norm_shader(self) -> str:
        """GLSL compute shader for layer normalization"""
        
        shader = """#version 450

// Qwen 32B Layer Normalization Shader for Radeon 780M
// Efficient layer norm with variance computation

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

layout(std430, binding = 1) readonly buffer WeightBuffer {
    float weight_data[];  // Layer norm weights
};

layout(std430, binding = 2) readonly buffer BiasBuffer {
    float bias_data[];    // Layer norm bias
};

layout(std430, binding = 3) writeonly buffer OutputBuffer {
    float output_data[];
};

layout(push_constant) uniform PushConstants {
    uint hidden_size;     // 5120
    float epsilon;        // 1e-5
};

shared float shared_sum[256];
shared float shared_sum_sq[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint idx = gl_GlobalInvocationID.x;
    uint elements_per_thread = (hidden_size + 255) / 256;
    
    // Compute local sum and sum of squares
    float local_sum = 0.0;
    float local_sum_sq = 0.0;
    
    for (uint i = 0; i < elements_per_thread; i++) {
        uint data_idx = tid * elements_per_thread + i;
        if (data_idx < hidden_size) {
            float val = input_data[data_idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    
    barrier();
    
    // Reduce sums across workgroup
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        barrier();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / float(hidden_size);
    float variance = (shared_sum_sq[0] / float(hidden_size)) - (mean * mean);
    float inv_std = inversesqrt(variance + epsilon);
    
    // Apply layer normalization
    for (uint i = 0; i < elements_per_thread; i++) {
        uint data_idx = tid * elements_per_thread + i;
        if (data_idx < hidden_size) {
            float normalized = (input_data[data_idx] - mean) * inv_std;
            output_data[data_idx] = normalized * weight_data[data_idx] + bias_data[data_idx];
        }
    }
}
"""
        return shader
    
    def compile_shaders(self) -> Dict:
        """Compile GLSL shaders to SPIR-V"""
        
        logger.info("🔨 Compiling Vulkan GLSL shaders to SPIR-V...")
        
        compiled_shaders = {}
        
        for shader_name, shader_code in self.shader_templates.items():
            try:
                # Save shader to file
                shader_file = f"/tmp/qwen32b_{shader_name}.comp"
                with open(shader_file, 'w') as f:
                    f.write(shader_code)
                
                # Compile to SPIR-V
                spirv_file = f"/tmp/qwen32b_{shader_name}.spv"
                
                logger.info(f"   🔧 Compiling {shader_name}...")
                
                # Compilation command (requires glslangValidator)
                import subprocess
                try:
                    result = subprocess.run([
                        "glslangValidator", "-V", shader_file, "-o", spirv_file
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        compiled_shaders[shader_name] = {
                            "source": shader_file,
                            "spirv": spirv_file,
                            "compiled": True,
                            "size": os.path.getsize(spirv_file) if os.path.exists(spirv_file) else 0
                        }
                        logger.info(f"      ✅ {shader_name} compiled successfully")
                    else:
                        raise Exception(f"glslangValidator failed: {result.stderr}")
                        
                except FileNotFoundError:
                    # Fallback - create placeholder
                    compiled_shaders[shader_name] = {
                        "source": shader_file,
                        "spirv": spirv_file,
                        "compiled": False,
                        "error": "glslangValidator not found - placeholder created"
                    }
                    logger.warning(f"      ⚠️ {shader_name} - glslangValidator not found")
                
            except Exception as e:
                logger.error(f"      ❌ {shader_name} compilation failed: {e}")
                compiled_shaders[shader_name] = {
                    "source": None,
                    "spirv": None,
                    "compiled": False,
                    "error": str(e)
                }
        
        return compiled_shaders
    
    def optimize_for_radeon_780m(self) -> Dict:
        """Radeon 780M specific optimizations"""
        
        optimizations = {
            "workgroup_size": {
                "matrix_ops": [16, 16, 1],    # 16x16 for matrix operations
                "vector_ops": [256, 1, 1],    # 256 for vector operations
                "reduction_ops": [256, 1, 1]  # 256 for reductions
            },
            "memory_optimization": {
                "buffer_alignment": 256,       # 256-byte alignment for optimal access
                "local_memory_usage": 32768,   # 32KB local memory per workgroup
                "prefetch_strategy": "double_buffer",
                "cache_optimization": True
            },
            "precision_strategy": {
                "weights": "INT4_GROUPED",     # INT4 with group quantization
                "activations": "FP16",         # FP16 for activations
                "accumulation": "FP32",        # FP32 for accumulation
                "final_output": "INT8"         # INT8 for final output
            },
            "rdna3_features": {
                "wave64_mode": True,           # Use Wave64 for better occupancy
                "infinity_cache": True,        # Utilize Infinity Cache
                "async_compute": True,         # Overlap compute and memory
                "primitive_shaders": False     # Disable for compute workloads
            }
        }
        
        return optimizations
    
    def create_vulkan_pipeline(self) -> Dict:
        """Create Vulkan compute pipeline configuration"""
        
        pipeline = {
            "stages": [
                {
                    "name": "gate_projection",
                    "shader": "qwen32b_gate_projection.spv",
                    "dispatch_size": [1708, 1, 1],  # ceil(27392/16)
                    "input_buffers": ["hidden_states", "gate_weights", "gate_scales", "gate_zero_points"],
                    "output_buffers": ["gate_output"]
                },
                {
                    "name": "up_projection", 
                    "shader": "qwen32b_up_projection.spv",
                    "dispatch_size": [1708, 1, 1],
                    "input_buffers": ["hidden_states", "up_weights", "up_scales", "up_zero_points"],
                    "output_buffers": ["up_output"]
                },
                {
                    "name": "silu_activation",
                    "shader": "qwen32b_silu_activation.spv", 
                    "dispatch_size": [107, 1, 1],  # ceil(27392/256)
                    "input_buffers": ["gate_output", "up_output"],
                    "output_buffers": ["activated_output"]
                },
                {
                    "name": "down_projection",
                    "shader": "qwen32b_down_projection.spv",
                    "dispatch_size": [20, 1, 1],  # ceil(5120/256)
                    "input_buffers": ["activated_output", "down_weights", "down_scales", "down_zero_points"],
                    "output_buffers": ["ffn_output"]
                },
                {
                    "name": "layer_norm",
                    "shader": "qwen32b_layer_norm.spv",
                    "dispatch_size": [1, 1, 1],  # Single workgroup for layer norm
                    "input_buffers": ["ffn_output", "norm_weights", "norm_bias"],
                    "output_buffers": ["normalized_output"]
                }
            ],
            "synchronization": {
                "barriers": True,
                "timeline_semaphores": True,
                "pipeline_barriers": "memory_dependency"
            }
        }
        
        return pipeline

def main():
    """Test Vulkan FFN shader compilation"""
    
    logger.info("🦄 Qwen 2.5 32B Vulkan FFN Shader Compiler")
    logger.info("=" * 60)
    
    # Initialize shader compiler
    shader_compiler = Qwen32BVulkanFFNShaders()
    
    # Compile shaders
    compiled_shaders = shader_compiler.compile_shaders()
    
    # Get optimizations
    optimizations = shader_compiler.optimize_for_radeon_780m()
    
    # Create pipeline
    pipeline = shader_compiler.create_vulkan_pipeline()
    
    # Summary
    successful_shaders = sum(1 for s in compiled_shaders.values() if s["compiled"])
    total_shaders = len(compiled_shaders)
    
    logger.info("=" * 60)
    logger.info("🎯 VULKAN SHADER COMPILATION COMPLETE!")
    logger.info(f"✅ Successful: {successful_shaders}/{total_shaders} shaders")
    logger.info(f"🔧 Target hardware: AMD Radeon 780M (RDNA3)")
    logger.info(f"🎯 Precision: INT4 weights, FP16 activations")
    logger.info(f"⚡ Pipeline stages: {len(pipeline['stages'])}")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())