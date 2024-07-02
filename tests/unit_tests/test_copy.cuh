#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../../src/helpf.cuh"
#include "../../src/gpu_assert.cuh"

// Kernel wrapper for copy2 function
template <typename T>
__global__ void copy2Kernel(uint32_t n1,
                            T alpha1,
                            T *src1,
                            T *dst1,
                            uint32_t n2,
                            T alpha2,
                            T *src2,
                            T *dst2)
{
    copy2(n1, alpha1, src1, dst1, n2, alpha2, src2, dst2);
}

// Kernel wrapper function for copy3
template <typename T>
__global__ void copy3Kernel(uint32_t n1,
                            T alpha1,
                            T *d_src1,
                            T *d_dst1,
                            uint32_t n2,
                            T alpha2,
                            T *d_src2,
                            T *d_dst2,
                            uint32_t n3,
                            T alpha3,
                            T *d_src3,
                            T *d_dst3)
{
    copy3(n1, alpha1, d_src1, d_dst1, n2, alpha2, d_src2, d_dst2, n3, alpha3, d_src3, d_dst3);
}

// Test case for copy2 function for positive numbers
TEST(CopyFunctionTest, Coopy2Post)
{
    const int N1 = 10;
    const int N2 = 5;
    const int N = N1 + N2;

    float *d_src1, *d_dst1, *d_src2, *d_dst2;
    float h_src1[N1], h_dst1[N1], h_src2[N2], h_dst2[N2];

    // Initialize input arrays on host
    for (int i = 0; i < N1; ++i)
    {
        h_src1[i] = static_cast<float>(i + 1); // Example data
    }
    for (int i = 0; i < N2; ++i)
    {
        h_src2[i] = static_cast<float>(i + 1); // Example data
    }

    // Allocate device memory
    gpuErrchk(cudaMalloc((void **)&d_src1, N1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_dst1, N1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_src2, N2 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_dst2, N2 * sizeof(float)));

    // Copy input arrays from host to device
    gpuErrchk(cudaMemcpy(d_src1, h_src1, N1 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src2, h_src2, N2 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(64); // Adjust block size as needed
    dim3 gridDim(4);    // Adjust grid size as needed
    copy2Kernel<<<gridDim, blockDim>>>(N1, 2.0f, d_src1, d_dst1, N2, 1.5f, d_src2, d_dst2);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_dst1, d_dst1, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_dst2, d_dst2, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result for d_dst1
    for (int i = 0; i < N1; ++i)
    {
        EXPECT_NEAR(h_dst1[i], 2.0f * h_src1[i],0.0001);
    }

    // Verify the result for d_dst2
    for (int i = 0; i < N2; ++i)
    {
        EXPECT_NEAR(h_dst2[i], 1.5f * h_src2[i],0.0001);
    }

    // Clean up
    gpuErrchk(cudaFree(d_src1));
    gpuErrchk(cudaFree(d_dst1));
    gpuErrchk(cudaFree(d_src2));
    gpuErrchk(cudaFree(d_dst2));
}
// Write another testcase for negative numbers

// write another test case for large arrays

// Test case for copy3 function
TEST(CopyFunctionTest, Copy3Test)
{
    const int N1 = 10;
    const int N2 = 5;
    const int N3 = 7;
    const int N = N1 + N2 + N3;

    float *d_src1, *d_dst1, *d_src2, *d_dst2, *d_src3, *d_dst3;
    float h_src1[N1], h_dst1[N1], h_src2[N2], h_dst2[N2], h_src3[N3], h_dst3[N3];

    // Initialize input arrays on host
    for (int i = 0; i < N1; ++i)
    {
        h_src1[i] = static_cast<float>(i + 1); // Example data
    }
    for (int i = 0; i < N2; ++i)
    {
        h_src2[i] = static_cast<float>(i + 1); // Example data
    }
    for (int i = 0; i < N3; ++i)
    {
        h_src3[i] = static_cast<float>(i + 1); // Example data
    }

    // Allocate device memory
    gpuErrchk(cudaMalloc((void **)&d_src1, N1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_dst1, N1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_src2, N2 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_dst2, N2 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_src3, N3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_dst3, N3 * sizeof(float)));

    // Copy input arrays from host to device
    gpuErrchk(cudaMemcpy(d_src1, h_src1, N1 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src2, h_src2, N2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_src3, h_src3, N3 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(256); // Adjust block size as needed
    dim3 gridDim(1);    // Adjust grid size as needed
    copy3Kernel<<<gridDim, blockDim>>>(N1, 2.0f, d_src1, d_dst1, N2, 1.5f, d_src2, d_dst2, N3, 0.5f, d_src3, d_dst3);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
      gpuErrchk(cudaMemcpy(h_dst1, d_dst1, N1 * sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(h_dst2, d_dst2, N2 * sizeof(float), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(h_dst3, d_dst3, N3 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result for d_dst1
    for (int i = 0; i < N1; ++i)
    {
        EXPECT_NEAR(h_dst1[i], 2.0f * h_src1[i],0.0001);
    }

    // Verify the result for d_dst2
    for (int i = 0; i < N2; ++i)
    {
        EXPECT_NEAR(h_dst2[i], 1.5f * h_src2[i],0.0001);
    }

    // Verify the result for d_dst3
    for (int i = 0; i < N3; ++i)
    {
        EXPECT_NEAR(h_dst3[i], 0.5f * h_src3[i],0.0001);
    }

    // Clean up
      gpuErrchk(cudaFree(d_src1));
      gpuErrchk(cudaFree(d_dst1));
      gpuErrchk(cudaFree(d_src2));
      gpuErrchk(cudaFree(d_dst2));
      gpuErrchk(cudaFree(d_src3));
      gpuErrchk(cudaFree(d_dst3));
}

