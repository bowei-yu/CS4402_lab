// gpu rank sort https://pastebin.com/2Eb2Rk4R

#include <stdio.h>

__host__ void sortOnHost(int n, int * h_a, int * h_b){

    for(int i=0;i<n;i++){
        int rank = 0;
        for(int j=0;j<n;j++){
            if(h_a[i]>h_a[j])
                rank++;
        }
        h_b[rank] = h_a[i];
    }
}

__global__ void sortOnDevice1(int n, int * d_a, int * d_b){

    // iteration i computed by threadId
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadId<n){
        int rank = 0;
        for(int j=0;j<n;j++){
            if(d_a[threadId]>d_a[j])
                rank++;
        }
        d_b[rank] = d_a[threadId];

    }

}



__global__ void sortOnDevice2(int n, int * d_a, int * d_b){

    // iteration i computed by threadId
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadId<n){
        int rank = 0, elem = d_a[threadId];
        for(int j=0;j<n;j++){
            if(elem>d_a[j])
                rank++;
        }
        d_b[rank] = elem;

    }

}

// shared memeory
__global__ void sortOnDevice3(int n, int * d_a, int * d_b){

    // iteration i computed by threadId
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    // copy d_a to share_a
    extern __shared__ int share_a[];

    if(threadIdx.x < warpSize){
        for(int i=threadIdx.x;i<n;i+=warpSize)
            share_a[i] = d_a[i];
    }
    __syncthreads();

    if(threadId<n){
        int rank = 0, elem = share_a[threadId];
        for(int j=0;j<n;j++){
            if(elem>share_a[j])
                rank++;
        }
        d_b[rank] = elem;

    }

}


int main(int argc, char ** argv){

    int n = atoi(argv[1]), blockSize = atoi(argv[2]);

    // allocate the arrays and initialise h_a

    int * h_a = (int *)malloc(n*sizeof(int));
    int * h_b = (int *)malloc(n*sizeof(int));

    int * d_a, * d_b;
    cudaMalloc((void**)&d_a, n*sizeof(int));
    cudaMalloc((void**)&d_b, n*sizeof(int));

    // initialize array, remove duplicates
    for(int i=0;i<n;i++)h_a[i] = n-i;

    clock_t time1, time2;

    // copy h_a to d_a
    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);

    // device sort
    time1 = clock();
    int gridSize = (n + blockSize - 1)/blockSize;
    sortOnDevice1<<<gridSize, blockSize>>>(n, d_a, d_b);
    time2 = clock();
    printf("exec time global memory access = % lf\n", 1.0*(time2-time1)/CLOCKS_PER_SEC);

    // device sort
    time1 = clock();

    sortOnDevice2<<<gridSize, blockSize>>>(n, d_a, d_b);
    time2 = clock();
    printf("exec time better global memory access = % lf\n", 1.0*(time2-time1)/CLOCKS_PER_SEC);

    // device sort
    time1 = clock();

    sortOnDevice3<<<gridSize, blockSize, n*sizeof(int)>>>(n, d_a, d_b);
    time2 = clock();
    printf("exec time shared memory = % lf\n", 1.0*(time2-time1)/CLOCKS_PER_SEC);

    // copy d_b to h_b anf print it
    cudaMemcpy(h_b, d_b, n*sizeof(int), cudaMemcpyDeviceToHost);

    //for(int i=0;i<n;i++) printf("%d ", h_b[i]);

    cudaFree(d_a);cudaFree(d_b);

    return 0;

}
