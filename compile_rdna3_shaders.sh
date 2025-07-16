#!/bin/bash
# Compile RDNA3-optimized shaders

echo "🔧 Compiling RDNA3-optimized shaders..."

# Check for glslangValidator
if ! command -v glslangValidator &> /dev/null; then
    echo "❌ glslangValidator not found. Install with: sudo apt install glslang-tools"
    exit 1
fi

# Compile with RDNA3 optimizations
echo "Compiling rdna3_optimized.comp..."
glslangValidator -V rdna3_optimized.comp -o rdna3_optimized.spv \
    --target-env vulkan1.2 \
    --quiet

if [ $? -eq 0 ]; then
    echo "✅ RDNA3 shader compiled successfully"
    ls -la rdna3_optimized.spv
else
    echo "❌ Shader compilation failed"
    exit 1
fi

# Also create attention-specific shader
cat > rdna3_attention.comp << 'EOF'
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// RDNA3-optimized attention computation
// Uses Wave32 mode and subgroup operations

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Q { float q[]; };
layout(binding = 1) readonly buffer K { float k[]; };
layout(binding = 2) readonly buffer V { float v[]; };
layout(binding = 3) writeonly buffer Output { float outData[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;
} params;

shared float scores[1024];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint batch = gid / params.num_heads;
    uint head = gid % params.num_heads;
    
    // Compute attention scores
    float max_score = -1e10;
    
    for (uint i = tid; i < params.seq_len; i += 32) {
        float score = 0.0;
        
        // Dot product Q[seq] · K[i]
        for (uint d = 0; d < params.head_dim; ++d) {
            uint q_idx = batch * params.seq_len * params.num_heads * params.head_dim +
                        0 * params.num_heads * params.head_dim +
                        head * params.head_dim + d;
            
            uint k_idx = batch * params.seq_len * params.num_heads * params.head_dim +
                        i * params.num_heads * params.head_dim +
                        head * params.head_dim + d;
                        
            score += q[q_idx] * k[k_idx];
        }
        
        score *= params.scale;
        scores[i] = score;
        max_score = max(max_score, score);
    }
    
    // Reduce max across wave
    max_score = subgroupMax(max_score);
    
    barrier();
    
    // Softmax
    float sum_exp = 0.0;
    for (uint i = tid; i < params.seq_len; i += 32) {
        scores[i] = exp(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    sum_exp = subgroupAdd(sum_exp);
    
    barrier();
    
    // Output weighted sum
    for (uint d = tid; d < params.head_dim; d += 32) {
        float result = 0.0;
        
        for (uint i = 0; i < params.seq_len; ++i) {
            uint v_idx = batch * params.seq_len * params.num_heads * params.head_dim +
                        i * params.num_heads * params.head_dim +
                        head * params.head_dim + d;
                        
            result += (scores[i] / sum_exp) * v[v_idx];
        }
        
        uint out_idx = batch * params.seq_len * params.num_heads * params.head_dim +
                      0 * params.num_heads * params.head_dim +
                      head * params.head_dim + d;
                      
        outData[out_idx] = result;
    }
}
EOF

echo "Compiling rdna3_attention.comp..."
glslangValidator -V rdna3_attention.comp -o rdna3_attention.spv \
    --target-env vulkan1.2 \
    --quiet

if [ $? -eq 0 ]; then
    echo "✅ Attention shader compiled successfully"
    ls -la rdna3_attention.spv
else
    echo "❌ Attention shader compilation failed"
fi

echo "🎯 RDNA3 shaders ready for use"