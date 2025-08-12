// Kernel to perform GELU activation and element-wise multiplication.

inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

__kernel void activate_and_multiply(
    __global float* gate_proj, // Will be modified in-place to become the activated result
    __global const float* up_proj,
    const int num_elements
) {
    const int i = get_global_id(0);
    if (i < num_elements) {
        gate_proj[i] = gelu(gate_proj[i]) * up_proj[i];
    }
}
