// Phase 1.1: QKV Projection Fusion - Starting Point
// Goal: Fuse 3 separate GEMM operations into one kernel
// Expected speedup: ~2x from eliminating 2 kernel launches

// Original approach (3 kernel launches):
// Q = input @ W_q  
// K = input @ W_k
// V = input @ W_v

// Fused approach (1 kernel launch):
// [Q, K, V] = input @ [W_q, W_k, W_v]

__kernel void qkv_projection_fused(
    __global const float* input,      // [batch*seq_len, hidden_size]
    __global const float* W_qkv,      // [hidden_size, 3*hidden_size] - packed weights
    __global float* output,           // [batch*seq_len, 3*hidden_size] - packed QKV
    const int batch_seq_len,          // batch_size * seq_len
    const int hidden_size
) {
    // Thread indexing
    const int row = get_global_id(0);  // which token
    const int col = get_global_id(1);  // which element of QKV
    
    if (row >= batch_seq_len || col >= 3 * hidden_size) return;
    
    // Compute one element of output
    float sum = 0.0f;
    
    // Dot product of input row with weight column
    #pragma unroll 8
    for (int k = 0; k < hidden_size; k++) {
        sum += input[row * hidden_size + k] * W_qkv[k * (3 * hidden_size) + col];
    }
    
    output[row * (3 * hidden_size) + col] = sum;
}

// Optimized version with tiling and local memory
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void qkv_projection_fused_optimized(
    __global const float* input,      
    __global const float* W_qkv,      
    __global float* output,           
    const int batch_seq_len,          
    const int hidden_size,
    __local float* lds  // Local data share
) {
    const int TILE_SIZE = 16;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    
    // Each thread computes one element
    const int row = bx * TILE_SIZE + tx;
    const int col = by * TILE_SIZE + ty;
    
    // Accumulator for this thread's output
    float sum = 0.0f;
    
    // LDS pointers
    __local float* tile_input = lds;
    __local float* tile_weights = lds + TILE_SIZE * TILE_SIZE;
    
    // Loop over tiles
    for (int t = 0; t < hidden_size; t += TILE_SIZE) {
        // Cooperatively load input tile
        if (row < batch_seq_len && t + ty < hidden_size) {
            tile_input[tx * TILE_SIZE + ty] = input[row * hidden_size + t + ty];
        } else {
            tile_input[tx * TILE_SIZE + ty] = 0.0f;
        }
        
        // Cooperatively load weight tile
        if (t + tx < hidden_size && col < 3 * hidden_size) {
            tile_weights[tx * TILE_SIZE + ty] = W_qkv[(t + tx) * (3 * hidden_size) + col];
        } else {
            tile_weights[tx * TILE_SIZE + ty] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_input[tx * TILE_SIZE + k] * tile_weights[k * TILE_SIZE + ty];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (row < batch_seq_len && col < 3 * hidden_size) {
        output[row * (3 * hidden_size) + col] = sum;
    }
}

// Helper kernel to split QKV after projection
__kernel void split_qkv(
    __global const float* qkv_packed,  // [batch*seq_len, 3*hidden_size]
    __global float* Q,                 // [batch*seq_len, hidden_size]
    __global float* K,                 // [batch*seq_len, hidden_size]
    __global float* V,                 // [batch*seq_len, hidden_size]
    const int batch_seq_len,
    const int hidden_size
) {
    const int idx = get_global_id(0);
    
    if (idx < batch_seq_len * hidden_size) {
        const int row = idx / hidden_size;
        const int col = idx % hidden_size;
        
        Q[idx] = qkv_packed[row * 3 * hidden_size + col];
        K[idx] = qkv_packed[row * 3 * hidden_size + hidden_size + col];
        V[idx] = qkv_packed[row * 3 * hidden_size + 2 * hidden_size + col];
    }
}

// Phase 1.2: Attention Score + Softmax Fusion
__kernel void attention_score_softmax_fused(
    __global const float* Q,           // [num_heads, seq_len, head_dim]
    __global const float* K,           // [num_heads, seq_len, head_dim]
    __global float* attention_weights, // [num_heads, seq_len, seq_len]
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale                 // 1/sqrt(head_dim)
) {
    const int head = get_global_id(0);
    const int row = get_global_id(1);
    
    if (head >= num_heads || row >= seq_len) return;
    
    // Compute one row of attention scores
    float max_score = -INFINITY;
    
    // First pass: compute scores and find max
    for (int col = 0; col <= row; col++) {  // Causal mask
        float score = 0.0f;
        
        // Dot product Q[row] @ K[col]
        for (int d = 0; d < head_dim; d++) {
            score += Q[head * seq_len * head_dim + row * head_dim + d] * 
                     K[head * seq_len * head_dim + col * head_dim + d];
        }
        
        score *= scale;
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = score;
        max_score = fmax(max_score, score);
    }
    
    // Second pass: exp and sum
    float sum = 0.0f;
    for (int col = 0; col <= row; col++) {
        float exp_score = exp(attention_weights[head * seq_len * seq_len + row * seq_len + col] - max_score);
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = exp_score;
        sum += exp_score;
    }
    
    // Third pass: normalize
    for (int col = 0; col <= row; col++) {
        attention_weights[head * seq_len * seq_len + row * seq_len + col] /= sum;
    }
    
    // Zero out future positions
    for (int col = row + 1; col < seq_len; col++) {
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = 0.0f;
    }
}