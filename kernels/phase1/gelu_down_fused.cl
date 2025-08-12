// Down Projection Kernel (pure GEMM)

#define TILE_SIZE 16

__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
void gelu_down_fused(
    const int M, // Number of rows in input A (batch_seq_len)
    const int N, // Number of columns in output C (hidden_size)
    const int K, // Number of columns in input A (ff_dim)
    const __global float* A, // Input tensor [M, K] (activated_output from gelu_multiply_kernel)
    const __global float* B, // Weights tensor [K, N] (W_down)
    __global float* C        // Output tensor [M, N]
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    __local float aTile[TILE_SIZE][TILE_SIZE]; // Tile for matrix A
    __local float bTile[TILE_SIZE][TILE_SIZE]; // Tile for matrix B

    float acc = 0.0f; // Accumulator for the C element

    // Loop over tiles along the K dimension (inner product dimension)
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Calculate global indices for loading elements into tiles
        const int tiledRowA = globalRow; // Row in A is globalRow
        const int tiledColA = t * TILE_SIZE + localCol; // Column in A is part of K dimension

        const int tiledRowB = t * TILE_SIZE + localRow; // Row in B is part of K dimension
        const int tiledColB = globalCol; // Column in B is globalCol

        // Load element into tile for matrix A
        // A[globalRow * K + tiledColA]
        if (globalRow < M && tiledColA < K) {
            aTile[localRow][localCol] = A[globalRow * K + tiledColA];
        } else {
            aTile[localRow][localCol] = 0.0f;
        }

        // Load element into tile for matrix B
        // B[tiledRowB * N + globalCol]
        if (tiledRowB < K && globalCol < N) {
            bTile[localRow][localCol] = B[tiledRowB * N + globalCol];
        } else {
            bTile[localRow][localCol] = 0.0f;
        }

        // Synchronize work-items to ensure all data is loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform matrix multiplication on the tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += aTile[localRow][k] * bTile[k][localCol];
        }

        // Synchronize again before loading the next set of tiles
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the final accumulated result to the output matrix C
    // C[globalRow * N + globalCol]
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = acc;
    }
}