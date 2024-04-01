#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../../GLASS/glass.cuh"
#include "./../help_functions/print_debug.cuh"
#include "./../help_functions/nested_dissect.cuh"


__device__ const bool DEBUG = true;
__device__ const int BLOCK = 0;
__device__ const bool THREAD = 0;
namespace cgrps = cooperative_groups;


/** @brief Checks that two 1D arrays are the same
 * @param T *array_a - pointer to the first array
 * @param T *array_b - pointer to the second array
 * @param uint32_t size - number of elements in each array
 */


/** @brief builds the binary tree into an array form
 * @param nhorizon number of timesteps
 * @param levels the array that will hold binary tree structure
 */
__device__ void initializeBSTLevels(int nhorizon, int *levels)
{
  int depth = log2f(nhorizon);

  for (int i = 0; i < nhorizon / 2; i++)
  {
    levels[2 * i] = 0;
    levels[2 * i + 1] = -1;
  }

  bool toggle = false;
  int previous_index = -1;
  for (int i = 1; i < depth; i++)
  {
    for (int j = 0; j < nhorizon; j++)
    {
      if (levels[j] != i - 1)
      {
        continue;
      }
      if (toggle)
      {
        levels[(previous_index + j) / 2] = i;
        toggle = false;
      }
      else
      {
        previous_index = j;
        toggle = true;
      }
    }
  }
}

/** @brief Gets all numbers at a specific level of the binary tree
 * @param nhorizon Number of timesteps
 * @param levels The array that holds the binary tree structure
 * @param targetLevel The level for which to retrieve the numbers
 * @param result An array to store the values at the specified level
 * @return count Number of timesteps at @p targetLevel
 */
__device__ int getValuesAtLevel(int nhorizon, const int *levels, int targetLevel, int *result)
{
  // Initialize the result array to ensure it's "cleaned" before each call
  for (int i = 0; i < nhorizon; i++)
  {
    result[i] = -1; // Use -1 to idicate no results
  }

  int count = 0;

  for (int i = 0; i < nhorizon; i++)
  {
    if (levels[i] == targetLevel)
    {
      result[count++] = i;
    }
  }
  return count;
}

// @brief The rsLQR tester, the main function of the solver

template <typename T>
__global__ void test_Kernel(uint32_t nhorizon,
                            uint32_t ninputs,
                            uint32_t nstates,
                            T *d_Q_R,
                            T *d_q_r,
                            T *d_A_B,
                            T *d_d,
                            T *d_F_lambda,
                            T *d_F_state,
                            T *d_F_input)

{
    // block/threads initialization
    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t grid_dim = gridDim.x;

    // KKT constants
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;
    const uint32_t depth = log2f(nhorizon);

    // initialize shared memory
    extern __shared__ T s_temp[];
    T *s_Q_R = s_temp;
    T *s_q_r = s_Q_R + (cost_step)*nhorizon;
    T *s_A_B = s_q_r + (ninputs + nstates) * nhorizon;
    T *s_d = s_A_B + (dyn_step)*nhorizon;
    T *s_F_lambda = s_d + nstates * nhorizon;
    T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
    T *s_F_input = s_F_state + (states_sq * nhorizon * depth);
    T *s_nI = s_F_input + (depth * inp_states * nhorizon);
    int *s_levels = (int *)(s_nI + states_sq);
    int *s_tree_result = (int *)(s_levels + nhorizon);

    // move ram to shared
    copy2<float>(cost_step * nhorizon, 1, d_Q_R, s_Q_R, dyn_step * nhorizon, 1, d_A_B, s_A_B);
    copy2<float>((nstates + ninputs) * nhorizon, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
    copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
                 d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);
    block.sync(); // block or grid?
    initializeBSTLevels(nhorizon, s_levels);
    block.sync();

    //check_equalitytest
    

}