#!/bin/bash
# Compile INT8 shaders to SPIR-V

echo "Compiling INT8 shaders..."

# Check if glslangValidator is available
if ! command -v glslangValidator &> /dev/null; then
    echo "Error: glslangValidator not found. Install Vulkan SDK."
    exit 1
fi

# Compile matrix multiply INT8 shader
echo "Compiling matrix_multiply_int8.comp..."
glslangValidator -V matrix_multiply_int8.comp -o matrix_multiply_int8.spv

# Compile gate_up_silu_mul INT8 shader
echo "Compiling gate_up_silu_mul_int8.comp..."
glslangValidator -V gate_up_silu_mul_int8.comp -o gate_up_silu_mul_int8.spv

# Create a simple down projection INT8 shader too
cat > down_proj_int8.comp << 'EOF'
#version 450

// Enable INT8 extensions
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_8bit_storage : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Push constants
layout(push_constant) uniform PushConstants {
    uint batch_seq_len;
    uint intermediate_size;
    uint hidden_size;
} pc;

// Input buffer (FP32)
layout(std430, binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

// INT8 weight buffer
layout(std430, binding = 1) readonly buffer WeightBuffer {
    int8_t weight[];  // [hidden_size, intermediate_size]
};

// Scale factor
layout(std430, binding = 2) readonly buffer WeightScale {
    float weight_scale[];
};

// Output buffer
layout(std430, binding = 3) writeonly buffer OutputBuffer {
    float output_data[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= pc.batch_seq_len * pc.hidden_size) {
        return;
    }
    
    uint batch_pos = idx / pc.hidden_size;
    uint out_idx = idx % pc.hidden_size;
    
    // INT8 matrix multiplication
    int32_t accumulator = 0;
    
    for (uint i = 0; i < pc.intermediate_size; ++i) {
        float input_val = input_data[batch_pos * pc.intermediate_size + i];
        int8_t input_int8 = int8_t(clamp(input_val * 127.0 / 8.0, -128.0, 127.0));
        int8_t w = weight[out_idx * pc.intermediate_size + i];
        accumulator += int32_t(input_int8) * int32_t(w);
    }
    
    // Dequantize
    float scale = weight_scale[0];
    float input_scale = 8.0 / 127.0;
    float result = float(accumulator) * scale * input_scale;
    
    output_data[idx] = result;
}
EOF

echo "Compiling down_proj_int8.comp..."
glslangValidator -V down_proj_int8.comp -o down_proj_int8.spv

echo "INT8 shaders compiled successfully!"
ls -la *.spv