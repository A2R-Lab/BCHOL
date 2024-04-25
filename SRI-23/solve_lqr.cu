#include <stdio.h>
#include <fstream>
#include <sstream>
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

template<typename T>
void read_csv(const std::string &filename, uint32_t &nhorizon, uint32_t &nstates, uint32_t &ninputs, T *Q_R, T *q_r, T *A_B, T *d) {
    // Open the CSV file
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read the CSV line
    std::string line;
    if (!getline(fin, line)) {
        std::cerr << "Error reading from file: " << filename << std::endl;
        fin.close();
        return;
    }

    // Close the file
    fin.close();

    // Parse the CSV line
    std::istringstream ss(line);
    std::string token;

    // Read nhorizon, nstates, and ninputs
    if (getline(ss, token, ',')) nhorizon = std::stoi(token);
    if (getline(ss, token, ',')) nstates = std::stoi(token);
    if (getline(ss, token, ',')) ninputs = std::stoi(token);

    // Check dimensions match
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;

    // Read Q_R
    for (int timestep = 0; timestep < nhorizon; ++timestep) {
        for (int i = 0; i < cost_step; ++i) {
            if (!getline(ss, token, ',')) {
                std::cerr << "Error reading Q_R from file: " << filename << std::endl;
                return;
            }
            Q_R[timestep * cost_step + i] = std::stod(token);
        }
    }

    // Read q_r
    for (int timestep = 0; timestep < nhorizon; ++timestep) {
        for (int i = 0; i < nstates + ninputs; ++i) {
            if (!getline(ss, token, ',')) {
                std::cerr << "Error reading q_r from file: " << filename << std::endl;
                return;
            }
            q_r[timestep * (nstates + ninputs) + i] = std::stod(token);
        }
    }

    // Read A_B
    for (int timestep = 0; timestep < nhorizon; ++timestep) {
        for (int i = 0; i < dyn_step; ++i) {
            if (!getline(ss, token, ',')) {
                std::cerr << "Error reading A_B from file: " << filename << std::endl;
                return;
            }
            A_B[timestep * dyn_step + i] = std::stod(token);
        }
    }

    // Read d
    for (int timestep = 0; timestep < nhorizon; ++timestep) {
        for (int i = 0; i < nstates; ++i) {
            if (!getline(ss, token, ',')) {
                std::cerr << "Error reading d from file: " << filename << std::endl;
                return;
            }
            d[timestep * nstates + i] = std::stod(token);
        }
    }

    std::cout << "CSV file has been read successfully." << std::endl;
}

__host__ int main()
{
  printf("Run Test\n");
  // Info about LQR problem

  // uint32_t nhorizon = 8;
  // uint32_t depth = log2(8);
  // uint32_t nstates = 6;
  // uint32_t ninputs = 3;

  uint32_t nhorizon = 0;
  uint32_t depth = 0;
  uint32_t nstates = 0;
  uint32_t ninputs = 0;

  float Q_R[(nstates * nstates + ninputs * ninputs) * nhorizon];
  float q_r[(nstates + ninputs) * nhorizon];
  float A_B[(nstates * nstates + nstates * ninputs) * nhorizon];
  float d[nstates * nhorizon];
  
  read_csv("examples/csv_example2.csv", nhorizon, nstates, ninputs, Q_R, q_r, A_B, d);


// Prints out the values to confirm the read_csv worked as intended
  for (uint32_t i = 0; i < nhorizon; ++i)
    {
        for (uint32_t j = 0; j < (nstates * nstates + ninputs * ninputs); ++j)
        {
            std::cout << Q_R[i * (nstates * nstates + ninputs * ninputs) + j] << " ";
        }
        std::cout << std::endl;
    }
  // Print the updated values of nhorizon, nstates, and ninputs
  std::cout << "Updated nhorizon: " << nhorizon << std::endl;
  std::cout << "Updated nstates: " << nstates << std::endl;
  std::cout << "Updated ninputs: " << ninputs << std::endl;


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
  std::uint32_t blockSize = 2;
  std::uint32_t gridSize = 2; 

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
