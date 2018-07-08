//
// Created by heidies on 7/5/18.
//

#include <cuda_runtime.h>
#include <iostream>

using namespace std;


int main(int argc, char **argv){
    cout << "Starting... " << endl;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess){
        cout << "cudaGetDeviceCount returned " << int(error_id) << endl;
        cout << "-> " <<cudaGetErrorString(error_id) << endl;
        cout << "Result = FAIL" << endl;
        //exit(EXIT_FAILURE);
    }

    if (deviceCount == 0){
        cout << "There is no available device that support CUDA" << endl;
    }
    else{
        cout << "Deteced " << deviceCount <<" CUDA Capable device(s)" << endl;
    }

    int dev, driverVersion = 0, runtimeVersion = 0;
    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "Device " << dev << "\"" << deviceProp.name << "\"" << endl;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << "  CUDA Driver Version / Runtime Version                      " << driverVersion / 1000 << "." << (driverVersion %100) / 10 << "/" <<
         runtimeVersion / 1000 << "." << (runtimeVersion%100) / 10 << endl;
    cout << "  CUDA Capability Major/Minor version number:                " << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "  Total amount of global memory:                             " << (float)deviceProp.totalGlobalMem/(pow(1024.0, 3)) << " GBytes" <<
         "(" << (unsigned long long) deviceProp.totalGlobalMem << " bytes)" << endl;
    cout << "  GPU Clock rate:                                            " << deviceProp.clockRate * 1e-3f  << " MHz" << "(" <<
         deviceProp.clockRate * 1e-6f << " GHz)" << endl;
    cout << "  Memory Clock rate:                                         " << deviceProp.memoryClockRate * 1e-3f << " Mhz" << endl;
    cout << "  Memory Bus Width:                                          " << deviceProp.memoryBusWidth << "-bit" << endl;
    if (deviceProp.l2CacheSize)
        cout << "  L2 Cache Size:                                             " << deviceProp.l2CacheSize << " bytes" << endl;
    cout << "  Max Texture Dimension Size (x, y, z)         1D=(" << deviceProp.maxTexture1D << "), " << "2D=(" <<
         deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture2D[1] << "), " << "3D=(" << deviceProp.maxTexture3D[0] << ", " <<
         deviceProp.maxTexture3D[1] << ", " << deviceProp.maxTexture3D[2] << ")" << endl;
    cout << "  Max Layered Texture Size (dim) x layers         1D=(" << deviceProp.maxTexture1DLayered[0] << ") x " <<
         deviceProp.maxTexture1DLayered[1] << "2D=(" << deviceProp.maxTexture2DLayered[0] << ", " << deviceProp.maxTexture2DLayered[1] << ") x " <<
         deviceProp.maxTexture2DLayered[2] << endl;
    cout << "  Total amount of constant memory:                           " << deviceProp.totalConstMem << " bytes" << endl;
    cout << "  Total amount of shared memory per block:                   " << deviceProp.sharedMemPerBlock << " bytes" << endl;
    cout << "  Total number of registers available per block:             " << deviceProp.regsPerBlock << endl;
    cout << "  Warp size:                                                 " << deviceProp.warpSize << endl;
    cout << "  Number of multiprocessors:                                 " << deviceProp.multiProcessorCount << endl;
    cout << "  Maximum number of warps per multiprocessor:                " << deviceProp.maxThreadsPerMultiProcessor / 32 << endl;
    cout << "  Maximum number of threads per multiprocessor:              " << deviceProp.maxThreadsPerMultiProcessor << endl;
    cout << "  Maximum number of threads per block:                       " << deviceProp.maxThreadsPerBlock << endl;
    cout << "  Maximum sizes of each dimension of a block:                " << deviceProp.maxThreadsDim[0] << " x " <<
         deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl;
    cout << "  Maximum sizes of each dimension of a grid:                 " << deviceProp.maxGridSize[0] << " x " <<
         deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << endl;
    cout << "  Maximum memory pitch:                                      " << deviceProp.memPitch << " bytes" << endl;

    exit(EXIT_SUCCESS);

}

