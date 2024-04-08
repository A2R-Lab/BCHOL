#include <stdio.h>
#include <iostream>
#include <cmath>
#include "solve.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// #include "blockassert.cuh" //need to write!

template <typename T>
__host__ __device__ void printMatrixH(T *matrix, uint32_t rows, uint32_t cols)
{
  for (unsigned i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f  ", matrix[j * rows + i]);
    }
    printf("\n");
  }
}
__host__ int main()
{
  printf("Run Test\n");
  // Info about LQR problem

  uint32_t nhorizon = 8;
  uint32_t depth = log2(8);
  uint32_t nstates = 6;
  uint32_t ninputs = 3;

  // float x0[6] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}; //instead put it as d0
  float Q_R[(nstates * nstates + ninputs * ninputs) * nhorizon] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q0
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R0
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q1
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R1
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q2
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R2
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q3
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R3
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q4
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R4
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q5
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R5
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q6
                                                                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                                                   0.01, 0.00, 0.00, // R6
                                                                   0.00, 0.01, 0.00,
                                                                   0.00, 0.00, 0.01,
                                                                   10.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Q7
                                                                   0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
                                                                   0.00, 0.00, 0.00, // R7
                                                                   0.00, 0.00, 0.00,
                                                                   0.00, 0.00, 0.00}; // Q_R diagonal matrices - doesn't matter row/column order

  float q_r[(nstates + ninputs) * nhorizon] = {
      -2.0, -1.2, -0.4, 0.4, 1.2, 2.0,                                                             // q0
      -1.0, 0.0, 1.0,                                                                              // r0
      -4.0, -2.4, -0.8, 0.8, 2.4, 4.0,                                                             // q1
      -2.0, 0.0, 2.0,                                                                              // r1
      -6.0, -3.5999999999999996, -1.2000000000000002, 1.2000000000000002, 3.5999999999999996, 6.0, // q2
      -3.0, 0.0, 3.0,                                                                              // r2
      -8.0, -4.8, -1.6, 1.6, 4.8, 8.0,                                                             // q3
      -4.0, 0.0, 4.0,                                                                              // r3
      -10.0, -6.0, -2.0, 2.0, 6.0, 10.0,                                                           // q4
      -5.0, 0.0, 5.0,                                                                              // r4
      -12.0, -7.199999999999999, -2.4000000000000004, 2.4000000000000004, 7.199999999999999, 12.0, // q5
      -6.0, 0.0, 6.0,                                                                              // r5
      -14.0, -8.4, -2.8000000000000003, 2.8000000000000003, 8.4, 14.0,                             // q6
      -7.0, 0.0, 7.0,                                                                              // r6
      -160.0, -96.0, -32.0, 32.0, 96.0, 160.0                                                      // q7
                                            - 0.0,
      0.0, 0.0 // r7 -doesn't exist
  };           // vectors q_r

  // c = 1 (?)
  float A_B[(nstates * nstates + nstates * ninputs) * nhorizon] = {
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A0 or column of A0^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A0
      0.005000000000000001, 0.0, 0.0, // row of B0 or column of B0^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B0
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A1 or column of A1^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A1
      0.005000000000000001, 0.0, 0.0, // row of B1 or column of B1^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B1
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A2 or column of A2^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A2
      0.005000000000000001, 0.0, 0.0, // row of B2 or column of B2^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B2
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A3 or column of A3^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A3
      0.005000000000000001, 0.0, 0.0, // row of B3 or column of B3^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B3
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A4 or column of A4^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A4
      0.005000000000000001, 0.0, 0.0, // row of B4 or column of B4^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B4
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A5 or column of A5^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A5
      0.005000000000000001, 0.0, 0.0, // row of B5 or column of B5^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B5
      1.0, 0.0, 0.0, 0.1, 0.0, 0.0, // row of A6 or column of A6^T
      0.0, 1.0, 0.0, 0.0, 0.1, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   // end of A6
      0.005000000000000001, 0.0, 0.0, // row of B6 or column of B6^T
      0.0, 0.005000000000000001, 0.0,
      0.0, 0.0, 0.005000000000000001,
      0.1, 0.0, 0.0,
      0.0, 0.1, 0.0,
      0.0, 0.0, 0.1,                // end of B6
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row of A7or column of A7^T
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // end of A7
      0.0, 0.0, 0.0,                // row of B7or column of B7^T
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0 // end of B7
  };

  float d[nstates * nhorizon] = {
      1.0, -1.0, 2.0, -2.0, 3.0, -3.0,   // x0
      1.5, 1.5, 1.5, 1.5, 1.5, 1.5,      // d0
      3.0, 3.0, 3.0, 3.0, 3.0, 3.0,      // d1
      4.5, 4.5, 4.5, 4.5, 4.5, 4.5,      // d2
      6.0, 6.0, 6.0, 6.0, 6.0, 6.0,      // d3
      7.5, 7.5, 7.5, 7.5, 7.5, 7.5,      // d4
      9.0, 9.0, 9.0, 9.0, 9.0, 9.0,      // d5
      10.5, 10.5, 10.5, 10.5, 10.5, 10.5 // d6
  };

  float F_lambda[nstates * nstates * nhorizon * depth];
  float F_state[nstates * nstates * nhorizon * depth];
  for (uint32_t n = 0; n < nstates * nstates * nhorizon * depth; n++)
  {
    F_lambda[n] = 0;
    F_state[n] = 0;
  }

  float F_input[ninputs * nstates * nhorizon * depth];
  for (uint32_t n = 0; n < ninputs * nstates * nhorizon * depth; n++)
  {
    F_input[n] = 0;
  }

  // Allocate memory on the GPU for x0,Q_R,q_r, A_B, d,

  float *d_Q_R;
  cudaMalloc((void **)&d_Q_R, (nstates * nstates + ninputs * ninputs) * nhorizon * sizeof(float));

  float *d_q_r;
  cudaMalloc((void **)&d_q_r, (nstates + ninputs) * nhorizon * sizeof(float));

  float *d_A_B;
  cudaMalloc((void **)&d_A_B, (nstates * nstates + nstates * ninputs) * nhorizon * sizeof(float));

  float *d_d;
  cudaMalloc((void **)&d_d, nstates * nhorizon * sizeof(float));
  printf("Allocated memory\n");

  float *d_F_lambda;
  cudaMalloc((void **)&d_F_lambda, nstates * nstates * nhorizon * depth * sizeof(float));

  float *d_F_state;
  cudaMalloc((void **)&d_F_state, nstates * nstates * nhorizon * depth * sizeof(float));

  float *d_F_input;
  cudaMalloc((void **)&d_F_input, nstates * ninputs * nhorizon * depth * sizeof(float));

  // Copy the matrices from the host to the GPU memory
  // cudaMemcpy(d_x0, x0, 6 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q_R, Q_R, (nstates * nstates + ninputs * ninputs) * nhorizon * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_r, q_r, (nstates + ninputs) * nhorizon * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_B, A_B, (nstates * nstates + nstates * ninputs) * nhorizon * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d, nstates * nhorizon * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_F_lambda, F_lambda, nstates * nstates * nhorizon * depth * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_F_state, F_state, nstates * nstates * nhorizon * depth * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_F_input, F_input, nstates * ninputs * nhorizon * depth * sizeof(float), cudaMemcpyHostToDevice);

  // Launch CUDA kernel with block and grid dimensions
  // uint32_t info[] = {nhorizon,ninputs,nstates};
  std::uint32_t blockSize = 256;
  std::uint32_t gridSize = 8;
  uint32_t shared_mem = 5 * 2160 * sizeof(float);
  const void *kernelFunc = reinterpret_cast<const void *>(solve_Kernel_t<float>);
  void *args[] = {// prepare the kernel arguments
                  &nhorizon,
                  &ninputs,
                  &nstates,
                  &d_Q_R,
                  &d_q_r,
                  &d_A_B,
                  &d_d,
                  &d_F_lambda,
                  &d_F_state,
                  &d_F_input};
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaDeviceSynchronize();
  cudaLaunchCooperativeKernel(kernelFunc, gridSize, blockSize, args, shared_mem);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }
  else
  {
    printf("launched successfully\n");
  }
  cudaDeviceSynchronize();
  cudaStatus = cudaGetLastError(); // Check for memory usage errors
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }
  printf("done with cuda!\n");

  // here can either launch one Kernel and call all functions within it and use blocks (cprgs)
  // or can potentially launch a kernel per each big function (solve_leaf etc)

  // check for error flags
  int error_flag_host;
  cudaMemcpyFromSymbol(&error_flag_host, error_flag, sizeof(int));
  if (error_flag_host == 1)
  {
    // Handle the error appropriately, e.g., print an error message
    printf("Error: Invalid index detected in the CUDA kernel tree_array.\n");
  }
  // Copy back to the host
  cudaMemcpy(q_r, d_q_r, (nstates + ninputs) * nhorizon * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d, d_d, nstates * nhorizon * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Q_R, d_Q_R, (nstates * nstates + ninputs * ninputs) * nhorizon * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_B, d_A_B, (nstates * nstates + nstates * ninputs) * nhorizon * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(F_lambda, d_F_lambda, nstates * nstates * nhorizon * depth * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(F_state, d_F_state, nstates * nstates * nhorizon * depth * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(F_input, d_F_input, nstates * ninputs * nhorizon * depth * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("\nSolve Time:  %3.1f ms \n", time);

  if (true)
  {
    printf("CHECK FINAL RESULTS on host\n");
    for (unsigned i = 0; i < nhorizon; i++)
    {
      printMatrix(d + i * nstates, nstates, 1);
      printMatrix(q_r + (i * (ninputs + nstates)), nstates, 1);
      printMatrix(q_r + (i * (ninputs + nstates) + nstates), ninputs, 1);
    }
  }

  // Free allocated GPU memory
  cudaFree(d_Q_R);
  cudaFree(d_q_r);
  cudaFree(d_A_B);
  cudaFree(d_d);
  cudaFree(d_F_lambda);
  cudaFree(d_F_state);
  cudaFree(d_F_input);
}
