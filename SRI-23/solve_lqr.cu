#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cmath>
#include "solve.cuh"
#include "./help_functions/csv.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>


__host__ int main()
{
  printf("Run Test\n");
  // Declaration of LQR problem
  uint32_t nhorizon = 8;
  uint32_t nstates = 6;
  uint32_t ninputs = 3;
  float Q_R[(nstates * nstates + ninputs * ninputs) * nhorizon];
  float q_r[(nstates + ninputs) * nhorizon];
  float A_B[(nstates * nstates + nstates * ninputs) * nhorizon];
  float d[nstates * nhorizon];
  uint32_t soln_size = (nstates + nstates + ninputs) * nhorizon - ninputs;
  float soln[soln_size];
  float my_soln[nstates * nhorizon + (nstates + ninputs) * nhorizon - ninputs];

  // Reading the LQR problem
  read_csv("lqr_prob8.csv", nhorizon, nstates, ninputs, Q_R, q_r, A_B, d, soln);
  uint32_t depth = log2(nhorizon);

  // Creating Factorization
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
  // when increasing blocksize to 32 not working
  std::uint32_t blockSize = 32;
  std::uint32_t gridSize = 8;

  uint32_t shared_mem = 5 * 2160 * sizeof(float);

  const void *kernelFunc = reinterpret_cast<const void *>(solve_Kernel<float>);
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
  // Prepare for timing
  cudaEvent_t start, stop;
  float time;
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


for (uint32_t timestep = 0; timestep < nhorizon; ++timestep)
{
  for (uint32_t i = 0; i < nstates; ++i)
  {
    my_soln[timestep * (nstates + nstates + ninputs) + i] = d[timestep * nstates + i];
  }
  for (uint32_t i = 0; i < nstates + ninputs; ++i)
  {
    my_soln[timestep * (nstates + nstates + ninputs) + nstates + i] = q_r[timestep * (nstates + ninputs) + i];
  }
}

if (checkEquality(my_soln, soln, soln_size))
{
  printf("PASSED!\n");
}
else
{
  printf("Not Passed");
  printf("my_soln\n");
  printMatrix(my_soln, (nstates + nstates + ninputs) * 2, 1);
  printf("Soln\n");
  printMatrix(soln, (nstates + nstates + ninputs) * 2, 1);
}

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