// program to generate primes

#include <stdio.h>
#include <math.h>

using namespace std;

// methods to work with 
__host__ int isPrimeOnHost(long n){

    if(n==0 || n==1)return 0;
    if(n==2 || n==3)return 1;
    if(n%2==0)return 0;
    for(long d=3;d<=sqrt(n);d+=2)
        if(n%d==0)return 0;
    return 1;
}

__host__ void testingPrimeOnHost(long n, int * h_answer){
    for(long d=0;d<n;d++)
        h_answer[d] = isPrimeOnHost(d);
}

__device__  int isPrimeOnDevice(long n){

    if(n==0 || n==1)return 0;
    if(n==2 || n==3)return 1;
    if(n%3==0)return 0;
    for(long d=3;d*d<=n;d+=2)
        if(n%d==0)return 0;
    return 1;
}

__global__ void testingPrimeOnDevice(size_t n, int * d_answer){

    // one thread = one iteration
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadId<n)
       d_answer[threadId] = isPrimeOnDevice(threadId);
}

int main(int argc, char ** argv){

    long n = atol(argv[1]);
    int * h_answer, * d_answer;
    h_answer = (int *)malloc(n*sizeof(int))
    cudaMalloc((void**)&d_answer, n*sizeof(int));

    // call the kernel
    int blockSize = 256, gridSize = (n+255)/256;

    double time = clock();
    testingPrimeOnDevice<<<gridSize,blockSize>>>(n, d_answer);
    time = clock()-time;
    printf("Execution of device %lf with %d thread per block and %d blocks\n", (double)time/CLOCKS_PER_SEC, blockSize, gridSize);

    time = clock();
    testingPrimeOnHost(n, h_answer);
    time = clock()-time;
    printf("Execution of host %lf\n", (double)time/CLOCKS_PER_SEC);

    // move d_answer to h_answer
    cudaMemcpy(h_answer, d_answer, n*sizeof(int), cudaMemcpyDeviceToHost);

    //for(long i=0;i<n;i++)printf("%d \n", h_answer[i]);

    return 0;
}
