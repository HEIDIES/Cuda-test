//
// Created by heidies on 7/7/18.
//

#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, const int nx, const int ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;
    if(ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

void initialData(float* ip, unsigned long long size){
    time_t t;
    srand((unsigned)time(&t));

    for(unsigned long long i = 0; i < size; ++i){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6) * 1e+3;
}

int main(int argc, char **argv){
    int nx = 1 << 14;
    int ny = 1 << 14;
    unsigned long long size = nx * ny;
    size_t nBytes = size * sizeof(float);

    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);

    initialData(h_A, size);
    initialData(h_B, size);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);

    int blockdimx = 32;
    int blockdimy = 16;

    if(argc > 2){
        blockdimx = atoi(argv[1]);
        blockdimy = atoi(argv[2]);
    }

    dim3 block(blockdimx, blockdimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    double iStart, iElaps;
    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    cout << "sumMatrixOnGPU2D <<< (" << grid.x << ", " << grid.y << "), " << "(" << block.x << ", " << block.y << ") >>> " <<
         "elapsed " << iElaps << " ms" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}

