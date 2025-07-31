// Passthrough Down Projection Kernel for Debugging
// This kernel performs the down-projection GEMM without any activation.
// It reads the 'gate' part of the input and multiplies it by the down-projection weights.

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void passthrough_down_fused(
    __global const float* restrict gate_up_output, // Input: [batch_seq_len, 2 * ff_dim]
    __global const float* restrict W_down,         // Weights: [ff_dim, hidden_size]
    __global float* restrict output,               // Output: [batch_seq_len, hidden_size]
    const int batch_seq_len,
    const int hidden_size,
    const int ff_dim
) {
    const int TILE_SIZE = 16;

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);

    const int global_row = group_row * TILE_SIZE + local_row;
    const int global_col = group_col * TILE_SIZE + local_col;

    __local float input_tile[TILE_SIZE][TILE_SIZE];
    __local float weights_tile_transposed[TILE_SIZE][TILE_SIZE];

    float accumulator = 0.0f;

    for (int k_tile_base = 0; k_tile_base < ff_dim; k_tile_base += TILE_SIZE) {
        
        // Load input tile (only the 'gate' part)
        const int load_col = k_tile_base + local_col;
        if (global_row < batch_seq_len && load_col < ff_dim) {
            input_tile[local_row][local_col] = gate_up_output[global_row * (2 * ff_dim) + load_col];
        } else {
            input_tile[local_row][local_col] = 0.0f;
        }

        // Load weights tile (transposed)
        const int weights_load_row = k_tile_base + local_row;
        const int weights_load_col = global_col;
        if (weights_load_row < ff_dim && weights_load_col < hidden_size) {
            weights_tile_transposed[local_col][local_row] = W_down[weights_load_row * hidden_size + weights_load_col];
        } else {
            weights_tile_transposed[local_col][local_row] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            accumulator += input_tile[local_row][k] * weights_tile_transposed[local_col][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < batch_seq_len && global_col < hidden_size) {
        output[global_row * hidden_size + global_col] = accumulator;
    }
}
