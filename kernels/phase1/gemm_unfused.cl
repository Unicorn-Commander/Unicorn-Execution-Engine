// A simple, unfused GEMM kernel for baseline comparison.

__kernel void gemm_unfused(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M, // A rows
    const int N, // B cols
    const int K  // A cols / B rows
) {
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    if (global_row >= M || global_col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[global_row * K + k] * B[k * N + global_col];
    }

    C[global_row * N + global_col] = acc;
}