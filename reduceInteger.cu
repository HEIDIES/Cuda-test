//
// Created by heidies on 7/8/18.
//
#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

#define CHECK(call)                                                                 \
{                                                                                  \
    const cudaError_t error = call;                                                   \
    if(error != cudaSuccess){                                                        \
        printf("Error: %s %d, ", __FILE__, __LINE__);                                \
        printf("code: %d, reason %s\n", error, cudaGetErrorString(error));              \
        exit(1);                                                                    \
    }                                                                              \
}


int recursiveReduce(int *data, const int size){
    if(size == 1) return data[0];
    int const stride = size / 2;

    for(int i = 0; i < stride; i++){
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}


__global__ void warmingUp(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return ;

    for(int stride = 1; stride < blockDim.x; stride <<= 1){
        if(tid % (2 * stride) == 0)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return ;

    for(int stride = 1; stride < blockDim.x; stride <<= 1){
        if(tid % (2 * stride) == 0)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return;

    for(int stride = 1; stride < blockDim.x; stride <<= 1){
        int index = 2 * stride * tid;
        if(index < blockDim.x / 2)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return;

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (2 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (2 * blockIdx.x) * blockDim.x;

    if(idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (4 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (4 * blockIdx.x) * blockDim.x;

    if(idx + 3 * blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (8 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (8 * blockIdx.x) * blockDim.x;

    if(idx + 7 * blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (8 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (8 * blockIdx.x) * blockDim.x;

    if(idx + 7 * blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1){
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (8 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (8 * blockIdx.x) * blockDim.x;

    if(idx + 7 * blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();

    if(blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64)
        idata[idx] += idata[tid + 64];
    __syncthreads();

    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = (8 * blockIdx.x) * blockDim.x + threadIdx.x;

    int *idata = g_idata + (8 * blockIdx.x) * blockDim.x;

    if(idx + 7 * blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2 * blockDim.x];
        g_idata[idx] += g_idata[idx + 3 * blockDim.x];
        g_idata[idx] += g_idata[idx + 4 * blockDim.x];
        g_idata[idx] += g_idata[idx + 5 * blockDim.x];
        g_idata[idx] += g_idata[idx + 6 * blockDim.x];
        g_idata[idx] += g_idata[idx + 7 * blockDim.x];
    }
    __syncthreads();

    if(iBlockSize >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if(iBlockSize >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if(iBlockSize >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if(iBlockSize >= 128 && tid < 64)
        idata[idx] += idata[tid + 64];
    __syncthreads();

    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6) * 1e+3;
}


int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "Starting reduction at " << argv[0] << " ";
    cout << "device " << dev << ": " << deviceProp.name << " ";
    cudaSetDevice(dev);

    bool bResult = false;

    int size = 1 << 24;
    cout << "   with array size " << size << "   ";

    int blocksize = 512;
    if(argc > 1)
        blocksize = atoi(argv[1]);

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);

    cout << "grid " << grid.x << " block " << block.x << endl;

    size_t nBytes = size * sizeof(int);
    int *h_idata = (int *)malloc(nBytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(nBytes);

    for(int i = 0; i < size; ++ i){
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, nBytes);

    double iStart, iElaps;
    int gpu_sum = 0;
    int *d_idata = NULL;
    cudaMalloc((void **)&d_idata, nBytes);
    int *d_odata = NULL;
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    cout << "cpu reduce     elapsed " << iElaps << " ms cpu_sum: " << cpu_sum << endl;


    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingUp<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for(int i = 0; i < grid.x; ++ i){
        gpu_sum += h_odata[i];
    }
    cout << "gpu warmingUp elapsed " << iElaps << " ms gpu_sum: " << gpu_sum << " <<<grid " << grid.x << " block " << block.x << ">>>" << endl;

    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for(int i = 0; i < grid.x; ++ i){
        gpu_sum += h_odata[i];
    }
    cout << "gpu Neighbored elapsed " << iElaps << " ms gpu_sum: " << gpu_sum << " <<<grid " << grid.x << " block " << block.x << ">>>" << endl;

    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for(int i = 0; i < grid.x / 8; ++ i){
        gpu_sum += h_odata[i];
    }
    cout << "gpu nroll elapsed " << iElaps << " ms gpu_sum: " << gpu_sum << " <<<grid " << grid.x / 8 << " block " << block.x << ">>>" << endl;

    free(h_idata);
    free(h_odata);
    free(tmp);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) cout << "Test failed!" << endl;
    return EXIT_SUCCESS;
}