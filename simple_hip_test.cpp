#include <hip/hip_runtime.h>

__global__ void add_kernel(int* a, int* b, int* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    // This is just a dummy main to make it a valid compilation unit
    return 0;
}
