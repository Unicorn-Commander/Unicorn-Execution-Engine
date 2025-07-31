// Fixed MLP Fusion Kernels - Handling 2*ff_dim correctly
// Phase 1.3: MLP Block Fusion with proper dimension handling

// Kernel 1: Fused Gate and Up projections
// Input: [M, hidden_size] @ [hidden_size, ff_dim] -> [M, ff_dim] (gate)
// Input: [M, hidden_size] @ [hidden_size, ff_dim] -> [M, ff_dim] (up)
// Output: [M, 2*ff_dim] (concatenated gate and up)

__kernel void gate_up_fused_fixed(
    __global const float* restrict input,     // [M, hidden_size]
    __global const float* restrict W_gate,    // [hidden_size, ff_dim]
    __global const float* restrict W_up,      // [hidden_size, ff_dim]
    __global float* restrict output,          // [M, 2*ff_dim] - gate and up concatenated
    const int M,
    const int hidden_size,
    const int ff_dim
) {
    const int TILE_SIZE = 16;
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= M || col >= 2 * ff_dim) return;
    
    // Determine which projection we're computing
    const int is_up = (col >= ff_dim) ? 1 : 0;
    const int weight_col = is_up ? (col - ff_dim) : col;
    
    // Select appropriate weight matrix
    __global const float* W = is_up ? W_up : W_gate;
    
    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < hidden_size; k++) {
        sum += input[row * hidden_size + k] * W[k * ff_dim + weight_col];
    }
    
    // Write to correct position in output
    output[row * (2 * ff_dim) + col] = sum;
}

// Kernel 2: GELU activation and element-wise multiply
// Input: [M, 2*ff_dim] containing gate and up values
// Output: [M, ff_dim] = GELU(gate) * up

__kernel void gelu_multiply_fixed(
    __global const float* restrict gate_up,   // [M, 2*ff_dim]
    __global float* restrict output,          // [M, ff_dim]
    const int M,
    const int ff_dim
) {
    const int idx = get_global_id(0);
    
    if (idx >= M * ff_dim) return;
    
    const int row = idx / ff_dim;
    const int col = idx % ff_dim;
    
    // Extract gate and up values
    float gate_val = gate_up[row * (2 * ff_dim) + col];
    float up_val = gate_up[row * (2 * ff_dim) + ff_dim + col];
    
    // GELU approximation: x * sigmoid(1.702 * x)
    float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_val));
    float gelu_gate = gate_val * sigmoid;
    
    // Multiply and store
    output[row * ff_dim + col] = gelu_gate * up_val;
}

// Kernel 3: Down projection
// Input: [M, ff_dim] @ [ff_dim, hidden_size] -> [M, hidden_size]

__kernel void down_projection_fixed(
    __global const float* restrict input,     // [M, ff_dim]
    __global const float* restrict W_down,    // [ff_dim, hidden_size]
    __global float* restrict output,          // [M, hidden_size]
    const int M,
    const int ff_dim,
    const int hidden_size
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= M || col >= hidden_size) return;
    
    float sum = 0.0f;
    for (int k = 0; k < ff_dim; k++) {
        sum += input[row * ff_dim + k] * W_down[k * hidden_size + col];
    }
    
    output[row * hidden_size + col] = sum;
}

// Alternative: Single kernel MLP fusion (if memory permits)
// This avoids the 2*ff_dim intermediate buffer entirely

__kernel void mlp_fused_single(
    __global const float* restrict input,     // [M, hidden_size]
    __global const float* restrict W_gate,    // [hidden_size, ff_dim]
    __global const float* restrict W_up,      // [hidden_size, ff_dim]
    __global const float* restrict W_down,    // [ff_dim, hidden_size]
    __global float* restrict output,          // [M, hidden_size]
    __local float* restrict lds,              // Local memory for tiling
    const int M,
    const int hidden_size,
    const int ff_dim
) {
    const int TILE_SIZE = 16;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    
    const int row = bx * TILE_SIZE + tx;
    const int col = by * TILE_SIZE + ty;
    
    if (row >= M || col >= hidden_size) return;
    
    // For each output element, we need to compute:
    // output[row,col] = sum_k( GELU(gate[row,k]) * up[row,k] * W_down[k,col] )
    
    float accumulator = 0.0f;
    
    // Process in chunks to fit in registers
    for (int k_start = 0; k_start < ff_dim; k_start += TILE_SIZE) {
        // Compute gate and up values for this tile
        float gate_tile[TILE_SIZE];
        float up_tile[TILE_SIZE];
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE && k_start + k < ff_dim; k++) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            
            // Compute gate[row, k_start+k] and up[row, k_start+k]
            for (int j = 0; j < hidden_size; j++) {
                float input_val = input[row * hidden_size + j];
                gate_sum += input_val * W_gate[j * ff_dim + k_start + k];
                up_sum += input_val * W_up[j * ff_dim + k_start + k];
            }
            
            // Apply GELU to gate
            float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_sum));
            gate_tile[k] = gate_sum * sigmoid;
            up_tile[k] = up_sum;
        }
        
        // Now compute contribution to output
        #pragma unroll
        for (int k = 0; k < TILE_SIZE && k_start + k < ff_dim; k++) {
            accumulator += gate_tile[k] * up_tile[k] * W_down[(k_start + k) * hidden_size + col];
        }
    }
    
    output[row * hidden_size + col] = accumulator;
}