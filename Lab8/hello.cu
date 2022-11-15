#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello from thread %d/%d in block %d/%d\n", threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
}

__host__ void host_hello() {
    printf("Hello from host\n");
}

int main() {
    int blockSize = 256;
    int numBlocks = 1;

    cuda_hello<<<numBlocks, blockSize>>>();
    cudaDeviceSynchronize();
    
    host_hello();

    return 0;
}