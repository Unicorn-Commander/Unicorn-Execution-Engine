// INT4 GEMM Kernel for RDNA3 (gfx1103)
// Unpacks two 4-bit integers from a single byte

#define BLOCK_SIZE 32 // Wave32 optimization

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void gemm_int4_ultra_speed(
    __global const uint* restrict A_packed,
    __global const uint* restrict B_packed,
    __global float* restrict C,
    const int M, const int N, const int K) {

    __local short A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local short B_tile[BLOCK_SIZE][BLOCK_SIZE];

    const int row = get_group_id(1) * BLOCK_SIZE + get_local_id(1);
    const int col = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    int sum_int = 0;

    // Tiled computation
    for (int k_base = 0; k_base < K; k_base += BLOCK_SIZE) {
        // Load tiles cooperatively
        // Coalesced load for A_tile
        if (row < M && (k_base + tx) < K) {
            uint packed_val = A_packed[row * (K / 8) + (k_base + tx) / 8];
            int shift = ((k_base + tx) % 8) * 4;
            A_tile[ty][tx] = (short)((packed_val >> shift) & 0x0F);
        } else {
            A_tile[ty][tx] = 0;
        }

        // Coalesced load for B_tile
        if ((k_base + ty) < K && col < N) {
            uint packed_val = B_packed[(k_base + ty) * (N / 8) + col / 8];
            int shift = ((col) % 8) * 4;
            B_tile[ty][tx] = (short)((packed_val >> shift) & 0x0F);
        } else {
            B_tile[ty][tx] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform matrix multiplication on tiles using integer arithmetic
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum_int += A_tile[ty][i] * B_tile[i][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result, converting to float at the end
    if (row < M && col < N) {
        C[row * N + col] = (float)sum_int;
    }
}