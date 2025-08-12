#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

__global__ void gemm_int4_wmma_kernel(
    const uint8_t* A_packed,  // INT4 packed (2 values per byte)
    const uint8_t* B_packed,  // INT4 packed
    int32_t* C,
    int M, int N, int K
) {
    // Use 16x16x16 tiles
    const int warpM = 16;
    const int warpN = 16;
    const int warpK = 16;
    
    // Calculate tile position
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x % 32;
    
    // Load matrix fragments using rocWMMA
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, uint8_t, rocwmma::col_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, uint8_t, rocwmma::row_major> b_frag;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> c_frag;
    
    // Initialize accumulator
    rocwmma::fill_fragment(c_frag, 0);
    
    // Perform WMMA operation
    for (int k = 0; k < K; k += warpK) {
        // Load A and B fragments
        rocwmma::load_matrix_sync(a_frag, A_packed + ..., M);
        rocwmma::load_matrix_sync(b_frag, B_packed + ..., K);
        
        // Perform INT4 WMMA
        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    rocwmma::store_matrix_sync(C + ..., c_frag, N, rocwmma::mem_row_major);
}