#pragma once

#include <iostream>
#include "test_chol.cuh"
#include <cuda_runtime.h>
#include "../help_functions/chol_InPlace.cuh"

// // Define a function to invoke the kernel with given arguments
// template <typename T>
// void test_chol_InPlace_1block(uint32_t n, T *s_A)
// {
//     // Define block and grid dimensions
//     dim3 blockDim(32, 1, 1); // Assuming 32x32 thread block
//     dim3 gridDim(1, 1, 1);   // Assuming 1x1 grid

//     // Invoke the kernel
//     chol_InPlace<<<gridDim, blockDim>>>(n, s_A);
//     cudaDeviceSynchronize();
// }


// Define a unit test function
void test_chol_InPlace()
{
    uint32_t n = 3;
    float h_A[n * n] = {
        6.0f, 15.0f, 55.0f,
        15.0f, 55.0f, 225.0f,
        55.0f, 225.0f, 979.0f};
    float *d_A;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    // copy matrices from host to GPU
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    // Define block and grid dimensions
    std::uint32_t blockSize = 1;
    std::uint32_t gridSize = 1;
    uint32_t shared_mem = 5 * 2160 * sizeof(float);

    // Invoke the kernel
    const void *kernelFunc = reinterpret_cast<const void *>(test_chol<float>);
    void *args[] = {// prepare the kernel arguments
                    &n,
                    &h_A};
    cudaDeviceSynchronize();
    cudaLaunchCooperativeKernel(kernelFunc, gridSize, blockSize, args, shared_mem);
    cudaDeviceSynchronize();
    // Copy the result back to host memory
    cudaMemcpy(h_A, d_A, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Add assertions
    // For simplicity, print the result for now
    std::cout << "Resultant matrix after Cholesky decomposition:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << h_A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);

    //ADD tests for 1 thread,1 block

    //ADD tests for many threads many blocks
}
