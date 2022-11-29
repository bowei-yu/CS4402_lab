#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <math.h>

using namespace std;

__host__ void reductionOnHost(int n, int * a, int * sum) {
    // calculate the sum of a
    * sum = 0;
    for (int i = 0; i< n; i++) {
        // line has data dependency, so no one thread one iteration. must use atomic operation
        *sum += a[i];
    }
}

__global__ void reductionNaiveOnDevice(int n, int * a, int * b) {
    // move a to shared memory
    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for (int i = threadIdx.x; i < n; i+=warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();

    // each thread computes a partial sum in a cyclic fashion
    int sum = 0;
    for (int i = threadIdx.x; i < n; i+=blockDim.x) {
        sum += shared_a[i];
    }
    __syncthreads();


    // atomic sum
    atomicSum(b, sum);
    __syncthreads();
}

__global__ void reductionBinaryOnDevice1(int n, int * a, int * b) {
    // move a to shared memory
    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for (int i = threadIdx.x; i < n; i+=warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();

    // each thread computes a partial sum in a cyclic fashion
    int sum = 0;
    for (int i = threadIdx.x; i < n; i+=blockDIm.x) {
        sum += shared_a[i];
    }
    __syncthreads();

    shared_a[threadIdx.x] = sum;

    // make a binary tree
    int threadId = threadIdx.x;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (threadId % (2 * stride) == 0) {
            shared_a[threadId] += shared_a[threadId + stride];
        }
    }
    __syncthreads();

    if (threadId == 0) {
        *b = shared_a[0];
    }
}

__global__ void reductionBinaryOnDevice2(int n, int * a, int * b) {
    // move a to shared memory
    extern __shared__ int shared_a[];

    if (threadIdx.x < warpSize) {
        for (int i = threadIdx.x; i < n; i+=warpSize) {
            shared_a[i] = a[i];
        }
    }
    __syncthreads();

    // each thread computes a partial sum in a cyclic fashion
    int sum = 0;
    for (int i = threadIdx.x; i < n; i+=blockDIm.x) {
        sum += shared_a[i];
    }
    __syncthreads();

    shared_a[threadIdx.x] = sum;

    // make a binary tree
    int threadId = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            shared_a[threadId] += shared_a[threadId + stride];
        }
    }
    __syncthreads();

    if (threadId == 0) {
        *b = shared_a[0];
    }
}

int main(in argv, char **argc) {

    int n = atoi(argc[1]), threads = atoi(argc[2]), blocks = 1;

    // allocate the arrays for host and device
    int * h_a = (int *) malloc(n*sizeof(int)), *h_b = malloc(sizeof(int));

    int * d_a, * d_b;
    cudaMalloc((void**) d_a, sizeof(int) * n);
    cudaMalloc((void**) d_b, sizeof(int));

    // initialize h_a and transfer it to d_a
    // initiate to 1 so we don't exceed the max integer
    for (int i = 0; i < n; i++) h_a[i] = 1;
    cudaMemcpy(d_a, h_a, n*sizeof(int, cudaMemcpyFromHostToDevice));

    // call the method to evaluate them
    clock_t time1 = clock();
    reductionNaiveOnDevice<<<blocks, threads>>>(n, d_a, d_b);
    clock_t time2 = clock();

    printf("Naive reduction: %lf\n", (double) (time2 - time1)/CLOCKS_PER_SEC);

    // call the method to evaluate them
    time1 = clock();
    reductionBinaryOnDevice1<<<blocks, threads>>>(n, d_a, d_b);
    time2 = clock();

    printf("Binary reduction with gaps: %lf\n", (double) (time2 - time1)/CLOCKS_PER_SEC);
    
    // call the method to evaluate them
    time1 = clock();
    reductionBinaryOnDevice2<<<blocks, threads>>>(n, d_a, d_b);
    time2 = clock();

    printf("Binary reduction with no gaps: %lf\n", (double) (time2 - time1)/CLOCKS_PER_SEC);

    // call the method to evaluate them
    time1 = clock();
    reductionOnHost(n, h_a, h_b);
    time2 = clock();

    printf("HOst reduction with gaps: %lf\n", (double) (time2 - time1)/CLOCKS_PER_SEC);

    return 0;
}



using namespace std;


__host__ void reductionOnHost(int n, int * a, int * sum){

        *sum = 0;
        for(int i=0;i<n;i++){
                *sum = *sum + a[i];
        }
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
        cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
        clock_t time2 = clock();

        printf("Naive reduction: Execution time on device %lf  \n", (double)(time2-time1)/CLOCKS_PER_SEC);

 
    time1 = clock();
    reductionBinaryOnDevice1<<<blocks, threads, blocks*sizeof(int)>>>(n, d_a, d_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
    time2 = clock();
    
    printf("Binary reduction 1: Execution time on device %lf  \n", (double)(time2-time1)/CLOCKS_PER_SEC);
    
    time1 = clock();
    reductionBinaryOnDevice2<<<blocks, threads, blocks*sizeof(int)>>>(n, d_a, d_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
    time2 = clock();
    
    printf("Binary reduction 2: Execution time on device %lf  \n", (double)(time2-time1)/CLOCKS_PER_SEC);
    
    time1 = clock();
    reductionOnHost(n, h_a, h_b);
   
    time2 = clock();
    
    printf("Reduction on Host: Execution time on device %lf  \n", (double)(time2-time1)/CLOCKS_PER_SEC);
    

        cudaFree(d_a);
        cudaFree(d_b);

        return 0;
}
