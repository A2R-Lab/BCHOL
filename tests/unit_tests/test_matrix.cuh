#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <iostream>
#include "../../src/helpf.cuh"
#include "../../src/gpu_assert.cuh"

// Kernel wrappers

//Wrapper for diag_Matrix from vector
template <typename T>
__global__ void diag_Matrix_set_v_kernel(std::uint32_t n, T *v_Q, T *m_Q)
{
    diag_Matrix_set_v(n, v_Q, m_Q, cgrps::this_thread_block());
}

//Wrapper for diag_Matrix from constant
template <typename T>
__global__ void diag_Matrix_set_kernel(std::uint32_t n, T c, T *m_Q)
{
    diag_Matrix_set(n, c, m_Q, cgrps::this_thread_block());
}

//Wrapper for dot_product.cuj
template <typename T>
__global__ void dot_product_kernel(std::uint32_t m,
                                   std::uint32_t n,
                                   std::uint32_t k,
                                   T alpha,
                                   T *A,
                                   T *B,
                                   T beta,
                                   T *C)
{
    dot_product(m, n, k, alpha, A, B, beta, C, cgrps::this_thread_block());
}
//Wrapper for lowerBackSub
template <typename T>
__global__ void lowerBackSub_InPlace_kernel(T *s_A, T *s_B, bool istransposed, int n, int m) {
    cgrps::thread_group g = cgrps::this_thread_block();
    lowerBackSub_InPlace(s_A, s_B, istransposed, n, m, g);
}

// Test case for diag_Matrix_set_v function
TEST(MatrixFunctionsTest, DiagMatrixSetVTest)
{
    const int N = 5;                               // Example size of the matrix
    float h_v[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // Example vector
    float h_m[N * N];                              // Resulting matrix to be checked

    float *d_v, *d_m;
    cudaMalloc((void **)&d_v, N * sizeof(float));
    cudaMalloc((void **)&d_m, N * N * sizeof(float));

    // Copy input vector from host to device
    cudaMemcpy(d_v, h_v, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel wrapper for diag_Matrix_set_v
    dim3 blockDim(256); // Adjust block size as needed
    dim3 gridDim(1);    // Adjust grid size as needed
    diag_Matrix_set_v_kernel<<<gridDim, blockDim>>>(N, d_v, d_m);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_m, d_m, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i == j)
            {
                EXPECT_NEAR(h_m[i * N + j], h_v[i],0.0001);
            }
            else
            {
                EXPECT_NEAR(h_m[i * N + j], 0.0f,0.0001); // Ensure non-diagonal elements are zero
            }
        }
    }

    // Clean up
    cudaFree(d_v);
    cudaFree(d_m);
}

// Test case for diag_Matrix_set function
TEST(MatrixFunctionsTest, DiagMatrixSetTest)
{
    const int N = 5;  // Example size of the matrix
    float c = 2.0f;   // Example value to set on diagonal
    float h_m[N * N]; // Resulting matrix to be checked

    float *d_m;
    cudaMalloc((void **)&d_m, N * N * sizeof(float));

    // Launch kernel wrapper for diag_Matrix_set
    dim3 blockDim(256); // Adjust block size as needed
    dim3 gridDim(1);    // Adjust grid size as needed
    diag_Matrix_set_kernel<<<gridDim, blockDim>>>(N, c, d_m);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_m, d_m, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i == j)
            {
                EXPECT_NEAR(h_m[i * N + j], c,0.0001);
            }
            else
            {
                EXPECT_NEAR(h_m[i * N + j], 0.0f,0.0001); // Ensure non-diagonal elements are zero
            }
        }
    }

    // Clean up
    cudaFree(d_m);
}

// Test case for dot_product function - WRITE!
// TEST(MatrixFunctionsTest, DotProductTest)
// {
  
// }

TEST(MatrixFUnctionsTest,LowerBackSubTest) {
    const int N = 3;  // Size of the matrix
    const int M = 1;  // Number of columns in matrix B

    // Example lower triangular matrix A (column-major order)
    float h_A[N*N] = {1.0f, 0.0f, 0.0f,
                      2.0f, 1.0f, 0.0f,
                      3.0f, 2.0f, 1.0f};

    // Example matrix B
    float h_B[N*M] = {6.0f, 5.0f, 4.0f};

    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, N* N * sizeof(float));
    cudaMalloc((void **)&d_B, N * M* sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * M* sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel directly in the test
    dim3 blockDim(64);  // Adjust block size as needed
    dim3 gridDim(4);  // Adjust grid size as needed
    lowerBackSub_InPlace_kernel<<<gridDim, blockDim>>>(d_A, d_B, false, N, M);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
    cudaMemcpy(h_B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected result after back substitution
    float expected_B[N*M] = {6.0f, -7.0f, 2.0f};

    // Verify the result
    for (int i = 0; i < N*M; ++i) {
        EXPECT_NEAR(h_B[i], expected_B[i],0.0001);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
}
