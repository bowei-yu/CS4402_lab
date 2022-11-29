#include <stdio.h>

__global__ void addArraysOnDevice(int n, int * d_a, int * d_b) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < n) d_a[threadId] = d_a[threadId] + d_b[threadId];
}

int main(int argc, char** argv) {

    int n = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    // allocate and initialize h_a and h_b arrays
    int * h_a = (int *) malloc(n * sizeof(int));
    int * h_b = (int *) malloc(n * sizeof(int));

    printf("h_a: \n");
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        printf("%d ", h_a[i]);
    }
    printf("\nh_b: \n");
    for (int i = 0; i < n; i++) {
        h_b[i] = n - i;
        printf("%d ", h_b[i]);
    }

    // declare and cudaMallocate the d_a and d_b arrays
    int * d_a, * d_b;
    cudaMalloc((void**) &d_a, n*sizeof(int));
    cudaMalloc((void**) &d_b, n*sizeof(int));
    // copy h_a and h_b into d_a and d_b respectively
    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(int), cudaMemcpyHostToDevice);

    // get grid size
    int gridSize = (n + blockSize - 1)/blockSize;
    addArraysOnDevice<<<gridSize, blockSize>>>(n, d_a, d_b);

    // copy d_a to h_a and print
    cudaMemcpy(h_a, d_a, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nresults: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_a[i]);
    }

    cudaFree(d_a); cudaFree(d_b);

    return 0;

}