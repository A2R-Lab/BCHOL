#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <iostream>
#include "../../src/helpf.cuh" 
#include "../../src/gpu_assert.cuh"

// Define Kernel wrappers

template <typename T>
__global__ void cholKernel(uint32_t n, T *d_A)
{
    cooperative_groups::thread_group g = cooperative_groups::this_thread_block();
    chol_InPlace(n, d_A, g);
}

template <typename T>
__global__ void copyMult(uint32_t n, T *d_A)
{
    cooperative_groups::thread_group g = cooperative_groups::this_thread_block();
    chol_InPlace(n, d_A, g);
}
// Test case for chol_InPlace function
TEST(HelpFunctionTest, CholInPlaceTest)
{
    const int N = 3;
    float *d_A;
    gpuErrchk(cudaMalloc((void **)&d_A, N * N * sizeof(float)));

    // Launch the kernel
    dim3 blockDim(64);
    dim3 gridDim(4);
    cholKernel<<<gridDim, blockDim>>>(N, d_A);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host

    float h_result[N * N];
    gpuErrchk(cudaMemcpy(h_result, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected result
    float h_expected[N * N] = {
        2.0f, 0.0f, 0.0f,
        6.0f, 1.0f, 0.0f,
        -8.0f, 5.0f, 3.0f};

    //Verify the result
    for (int i=0; i<N*N; ++i) {
        EXPECT_NEAR(h_result[i],h_expected[i],1e-5); //check if you wanna change 1e
    }

    gpuErrchk(cudaFree(d_A));
}


