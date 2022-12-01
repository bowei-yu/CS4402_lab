#include <stdio.h>
#include <stdlib.h>


using namespace std;


__host__ void reductionOnHost(int n, int * a, int * sum){

        *sum = 0;
        for(int i=0;i<n;i++) *sum += a[i];
} 

__global__ void reductionNaiveOnDevice(int n, int * a, int * b){


        extern __shared__ int shared_a[];

        if (threadIdx.x < warpSize) {
        for(int i = threadIdx.x; i<n; i += warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();
    
    
        int sum = 0;

        for(int i = threadIdx.x;i<n;i+=blockDim.x){
                sum += shared_a[i];
        }

        atomicAdd(b, sum);
        __syncthreads();
}


__global__ void reductionNaiveOnDevice1(int n, int * a, int * b){

    
    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for(int i = threadIdx.x; i<n; i += warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();
    
        
    int sum = 0;
    
    for(int i = threadIdx.x * n / blockDim.x;i<(threadIdx.x+1) * n / blockDim.x;i++){
        sum += shared_a[i];
    }
    
    atomicAdd(b, sum);
     __syncthreads();
}



__global__ void reductionBinaryOnDevice1(int n, int * a, int * b){

    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for(int i = threadIdx.x; i<n; i += warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();
    
    
    for(int i = threadIdx.x;i<n;i+=blockDim.x){
        shared_a[threadIdx.x] += shared_a[i];
    }
    __syncthreads();
        
    int tid = threadIdx.x;
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {

        if(tid % (2*s) == 0){
            shared_a[tid] += shared_a[tid + s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid == 0) *b = shared_a[0];
     __syncthreads();
}


__global__ void reductionBinaryOnDevice2(int n, int * a, int * b){

    printf("Thread %d finds sum \n", threadIdx.x);
    
    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for(int i = threadIdx.x; i<n; i += warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();
    
    
    for(int i = threadIdx.x;i<n;i+=blockDim.x){
        shared_a[threadIdx.x] += shared_a[i];
    }
    __syncthreads();
    
    int tid = threadIdx.x;
    // do reduction in shared mem
    
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            shared_a[tid] += shared_a[tid + s];
        }
        __syncthreads();
    }

    
    // write result for this block to global mem
    if(tid == 0) *b = shared_a[0];
     __syncthreads();
}




int main(int argv, char ** argc) {

        const int n = atoi(argc[1]);

        int blocks = 1, threads = atoi(argc[2]);
         
        int * h_a = (int *)malloc(n*sizeof(int)),
        *h_b = (int *)malloc(sizeof(int));

        for (int i = 0; i < n; ++i) h_a[i] = 1;

        int * d_a, * d_b;
        cudaMalloc((void **)&d_a, sizeof(int)*n);
        cudaMalloc((void **)&d_b, sizeof(int));

        cudaMemcpy(d_a, h_a, sizeof(int)*n, cudaMemcpyHostToDevice);

        clock_t time1 = clock();
        reductionNaiveOnDevice<<<blocks, threads, blocks*sizeof(int)>>>(n, d_a, d_b);
        cudaDeviceSynchronize();
        cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);;
        clock_t time2 = clock();

        printf("Naive reduction: Execution time on device %lf, sum: %d\n", (double)(time2-time1)/CLOCKS_PER_SEC, *h_b);

 
        time1 = clock();
        reductionBinaryOnDevice1<<<blocks, threads, blocks*sizeof(int)>>>(n, d_a, d_b);
        cudaDeviceSynchronize();
        cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
        time2 = clock();
    
        printf("Binary reduction 1: Execution time on device %lf, sum: %d\n", (double)(time2-time1)/CLOCKS_PER_SEC, *h_b);
    
        time1 = clock();
        reductionBinaryOnDevice2<<<blocks, threads, blocks*sizeof(int)>>>(n, d_a, d_b);
        cudaDeviceSynchronize();
        cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
        time2 = clock();
    
        printf("Binary reduction 2: Execution time on device %lf, sum: %d\n", (double)(time2-time1)/CLOCKS_PER_SEC, *h_b);
    
        time1 = clock();
        reductionOnHost(n, h_a, h_b);
        time2 = clock();
    
        printf("Reduction on Host: Execution time on device %lf, sum: %d\n", (double)(time2-time1)/CLOCKS_PER_SEC, *h_b);
    

        cudaFree(d_a);
        cudaFree(d_b);

        return 0;
}
