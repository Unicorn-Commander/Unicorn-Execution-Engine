// Fused QKV Projection Kernel - Correct Rectangular Tiled GEMM

#define TILE_SIZE 16

__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
void qkv_projection_fused(
    const int M, const int N, const int K,
    const __global float* A,
    const __global float* B,
    __global float* C
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    __local float aTile[TILE_SIZE][TILE_SIZE];
    __local float bTile[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        const int tiledRow = t * TILE_SIZE + localRow;
        const int tiledCol = t * TILE_SIZE + localCol;

        if (globalRow < M && tiledCol < K) {
            aTile[localRow][localCol] = A[globalRow * K + tiledCol];
        } else {
            aTile[localRow][localCol] = 0.0f;
        }

        if (tiledRow < K && globalCol < N) {
            bTile[localRow][localCol] = B[tiledRow * N + globalCol];
        } else {
            bTile[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += aTile[localRow][k] * bTile[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = acc;
    }
}
