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

  uint32_t nhorizon = 1;
    uint32_t depth = log2(nhorizon);

  if(nhorizon==1)
    depth = 1;
  uint32_t nstates = 6;
  uint32_t ninputs = 3;

  // float x0[6] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}; //instead put it as d0
  float Q_R[(nstates * nstates + ninputs * ninputs) * nhorizon] = {1.0,1.0,1.0,1.0,1.0,1.0, //Q0
                                                                    0.01,0.01,0.01 //R0
                                                                    }; // Q_R diagonal matrices - doesn't matter row/column order

  float q_r[(nstates + ninputs) * nhorizon] = { -2.0,-1.2,-0.4,0.4,1.2,2.0, //q0
                                                -1.0,0.0,1.0  //r0
                                                };           // vectors q_r

  // c = 1 
  float A_B[(nstates * nstates + nstates * ninputs) * nhorizon] = {
    1.0,0.0,0.0,0.0,0.0,0.0,
    0.0,1.0,0.0,0.0,0.0,0.0,
    0.0,0.0,1.0,0.0,0.0,0.0,
    0.1,0.0,0.0,1.0,0.0,0.0,
    0.0,0.1,0.0,0.0,1.0,0.0,
    0.0,0.0,0.1,0.0,0.0,1.0, //A0
    0.005000000000000001,0.0,0.0,0.1,0.0,0.0,
    0.0,0.005000000000000001,0.0,0.0,0.1,0.0,
    0.0,0.0,0.005000000000000001,0.0,0.0,0.1 //B0
  }; //A_B matrices

  float d[nstates * nhorizon] = {
      1.5,1.5,1.5,1.5,1.5,1.5 //d0
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
  std::uint32_t blockSize = 64;
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
