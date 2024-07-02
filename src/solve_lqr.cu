#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cmath>
#include "solve.cuh"
#include "helpf.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include "gpu_assert.cuh"


__host__ int main()
{
  printf("Run Test\n");
  // Declaration of LQR problem
  uint32_t knot_points = 8;
  uint32_t state_size = 6;
  uint32_t control_size = 3;
  uint32_t depth = log2(knot_points);

  // calculating the constants
  const uint32_t states_sq = state_size * state_size;
  const uint32_t controls_sq = control_size * control_size;
  const uint32_t states_p_controls = state_size * control_size;
  const uint32_t states_s_controls = state_size + control_size;
  const uint32_t fstates_size = states_sq * knot_points * depth;
  const uint32_t fcontrol_size = states_p_controls * knot_points * depth;

  const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq + controls_sq) * knot_points - controls_sq) * sizeof(float));
  const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq + states_p_controls) * (knot_points - 1) * sizeof(float));
  const uint32_t KKT_g_SIZE_BYTES = static_cast<uint32_t>(((state_size + control_size) * knot_points - control_size) * sizeof(float));
  const uint32_t KKT_c_SIZE_BYTES = static_cast<uint32_t>((state_size * knot_points) * sizeof(float));
  const uint32_t KKT_FSTATES_SIZE_BYTES = static_cast<uint32_t>(fstates_size * sizeof(float));
  const uint32_t KKT_FCONTROL_SIZE_BYTES = static_cast<uint32_t>(fcontrol_size * sizeof(float));

  // const uint32_t DZ_SIZE_BYTES = static_cast<uint32_t>((states_s_controls * knot_points - control_size) * sizeof(float));

  float Q_R[((states_sq + controls_sq) * knot_points - controls_sq)];
  float q_r[((state_size + control_size) * knot_points - control_size)];
  float A_B[(states_sq + states_p_controls) * (knot_points - 1)];
  float d[(state_size * knot_points)];
  uint32_t soln_size = (state_size + state_size + control_size) * knot_points - control_size;
  float soln[soln_size];

  float my_soln[soln_size];

  // // Reading the LQR problem
  read_csv("../exmpls/lqr_prob8.csv", knot_points, state_size, control_size, Q_R, q_r, A_B, d);

  // Creating Factorization
  float F_lambda[fstates_size];
  float F_state[fstates_size];
  for (uint32_t n = 0; n < fstates_size; n++)
  {
    F_lambda[n] = 0;
    F_state[n] = 0;
  }

  float F_input[fcontrol_size];
  for (uint32_t n = 0; n < fcontrol_size; n++)
  {
    F_input[n] = 0;
  }

  // Allocate memory on the GPU for x0,Q_R,q_r, A_B, d,

  float *d_Q_R, *d_q_r, *d_A_B, *d_d,
      *d_F_lambda, *d_F_state, *d_F_input;

  gpuErrchk(cudaMalloc((void **)&d_Q_R, KKT_G_DENSE_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_q_r, KKT_g_SIZE_BYTES));
  gpuErrchk((cudaMalloc((void **)&d_A_B, KKT_C_DENSE_SIZE_BYTES)));
  gpuErrchk(cudaMalloc((void **)&d_d, KKT_c_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_lambda, KKT_FSTATES_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_state, KKT_FSTATES_SIZE_BYTES));
  gpuErrchk(cudaMalloc((void **)&d_F_input, fcontrol_size * sizeof(float)));
  gpuErrchk(cudaPeekAtLastError());

  // Copy the matrices from the host to the GPU memory
  // cudaMemcpy(d_x0, x0, 6 * sizeof(float), cudaMemcpyHostToDevice);
  gpuErrchk(cudaMemcpy(d_Q_R, Q_R, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_q_r, q_r, KKT_g_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_A_B, A_B, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_d, d, KKT_c_SIZE_BYTES, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_F_lambda, F_lambda, KKT_FSTATES_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_F_state, F_state, KKT_FSTATES_SIZE_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_F_input, F_input, KKT_FCONTROL_SIZE_BYTES, cudaMemcpyHostToDevice));

  // Launch CUDA kernel with block and grid dimensions
  // find a way to automate number of threads and blocks
  std::uint32_t blockSize = 64;
  std::uint32_t gridSize = 8;

  //put it into a function
  uint32_t bchol_shared_mem_size = KKT_C_DENSE_SIZE_BYTES + KKT_G_DENSE_SIZE_BYTES + KKT_c_SIZE_BYTES + KKT_g_SIZE_BYTES +
                                   KKT_FCONTROL_SIZE_BYTES + KKT_FSTATES_SIZE_BYTES + KKT_FSTATES_SIZE_BYTES + (knot_points * 3 * sizeof(int))+500;

  std::cout << "shared_mem: " << bchol_shared_mem_size << std::endl;

  const void *bchol_kernelFunc = reinterpret_cast<const void *>(solve_BCHOL<float>);
  void *bchol_kernelArgs[] = {// prepare the kernel arguments
                              &knot_points,
                              &control_size,
                              &state_size,
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

  std::cout << "Launching blocks " << gridSize << " launching threads" << blockSize << "shared memory" << bchol_shared_mem_size << std::endl;
  if (DEBUG)
  {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming you're using device 0

    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;

    // Calculate max blocks for cooperative launch
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, bchol_kernelFunc, blockSize, bchol_shared_mem_size);
    int maxBlocks = numBlocksPerSm * prop.multiProcessorCount;
    std::cout << "Max blocks for cooperative kernel launch: " << maxBlocks << std::endl;
  }
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventRecord(start, 0));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaLaunchCooperativeKernel(bchol_kernelFunc, gridSize, blockSize, bchol_kernelArgs, bchol_shared_mem_size));
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());

  printf("done with cuda!\n");

  // Copy back to the host
  gpuErrchk(cudaMemcpy(q_r, d_q_r, KKT_g_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(d, d_d, KKT_c_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Q_R, d_Q_R, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(A_B, d_A_B, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(F_lambda, d_F_lambda, KKT_FSTATES_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(F_state, d_F_state, KKT_FSTATES_SIZE_BYTES, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(F_input, d_F_input, KKT_FCONTROL_SIZE_BYTES, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaEventRecord(stop, 0));
  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&time, start, stop));
  printf("\nSolve Time:  %3.1f ms \n", time);

  for (uint32_t timestep = 0; timestep < knot_points; ++timestep)
  {
    for (uint32_t i = 0; i < state_size; ++i)
    {
      my_soln[timestep * (state_size + state_size + control_size) + i] = d[timestep * state_size + i];
    }
    for (uint32_t i = 0; i < states_s_controls; ++i)
    {
      my_soln[timestep * (state_size + state_size + control_size) + state_size + i] = q_r[timestep * (states_s_controls) + i];
    }
  }

  // if (checkEquality(my_soln, soln, soln_size))
  // {
  //   printf("PASSED!\n");
  // }
  // else
  // {
  //   printf("Not Passed");
  //   printf("my_soln\n");
  //   printMatrix(my_soln, (state_size + state_size + control_size) * 2, 1);
  //   printf("Soln\n");
  //   printMatrix(soln, (state_size + state_size + control_size) * 2, 1);
  // }

  std::cout << "size " << soln_size << std::endl;
  printMatrix(my_soln, soln_size, 1);

  // Free allocated GPU memory
  gpuErrchk(cudaFree(d_Q_R));
  gpuErrchk(cudaFree(d_q_r));
  gpuErrchk(cudaFree(d_A_B));
  gpuErrchk(cudaFree(d_d));
  gpuErrchk(cudaFree(d_F_lambda));
  gpuErrchk(cudaFree(d_F_state));
  gpuErrchk(cudaFree(d_F_input));
}