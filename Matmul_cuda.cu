#include <opencv2/core/cuda/common.hpp>
#define BLOCK_SIZE 2
#include <iostream>
#include <sys/time.h>


using namespace std;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec) * 1.e-6;
}
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
// Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
// Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(1, 1);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    double iStart = cpuSecond();
    cout << "Using time: " <<iStart << endl;
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);


// Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    double iElaps = cpuSecond() - iStart;

// Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
// Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
// Each thread computes one element of Csub
// by accumulating results into Cvalue
    float Cvalue = 0;
// Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
// Loop over all the sub-matrices of A and B that are
// required to compute Csub
// Multiply each pair of sub-matrices together
// and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
// Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
// Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
// Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
// Load Asub and Bsub from device memory to shared memory
// Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
// Synchronize to make sure the sub-matrices are loaded
// before starting the computation
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

int main(){
    Matrix *A, *B, *C;
    A = (Matrix*)malloc(sizeof(Matrix));
    B = (Matrix*)malloc(sizeof(Matrix));
    C = (Matrix*)malloc(sizeof(Matrix));
    A->width = B->width = C->width = 2;
    A->height = B->height = C->height = 2;
    A->stride = B->stride = C->stride = 2;
    float A_array[4] = {1.0, 2.0, 3.0, 4.0};
    float B_array[4] = {2.0, 0.0, 0.0, 2.0};
    float C_array[4] = {0.0, 0.0, 0.0, 0.0};
    A->elements = A_array;
    B->elements = B_array;
    C->elements = C_array;
    //cout << C->elements[0] << C->elements[1] << C->elements[2] << C->elements[3] << endl;
    MatMul(*A, *B, *C);
    cout << C->elements[0] << C->elements[1] << C->elements[2] << C->elements[3] << endl;
    free(A);
    free(B);
    free(C);
    return 0;
}