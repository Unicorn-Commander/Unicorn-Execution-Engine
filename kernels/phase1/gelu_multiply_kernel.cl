// GELU and Element-wise Multiply Kernel

inline float sigmoid(float x) {
    // Clamp input to prevent overflow/underflow in exp
    x = clamp(x, -10.0f, 10.0f);
    return 1.0f / (1.0f + exp(-x));
}

inline float gelu(float x) {
    return x * sigmoid(1.702f * x); // Approximation: x * sigmoid(1.702 * x)
}

__kernel void gelu_multiply_kernel(
    __global const float* restrict gate_up_output, // Input: [batch_seq_len, 2 * ff_dim]
    __global float* restrict output,          // Output: [batch_seq_len, ff_dim]
    const int num_elements, // batch_seq_len * ff_dim
    const int ff_dim
) {
    const int gid = get_global_id(0);
    if (gid < num_elements) {
        const int row = gid / ff_dim;
        const int col = gid % ff_dim;

        float gate_val = gate_up_output[row * (2 * ff_dim) + col];
        float up_val = gate_up_output[row * (2 * ff_dim) + ff_dim + col];
        output[gid] = gelu(gate_val) * up_val;
    }
}