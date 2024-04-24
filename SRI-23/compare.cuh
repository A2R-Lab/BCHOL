#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
#include "./help_functions/nested_dissect.cuh"
// should we put into one header file?

// #include "./help_functions/copy_mult.cuh"
// #include "./help_functions/chol_InPlace.cuh"
// #include "./help_functions/chol_SolveInPlace.cuh"
// #include "./help_functions/diag_Matrix_set.cuh"
// #include "./help_functions/set_const.cuh"
// #include "./help_functions/dot_product.cuh"
// #include "./help_functions/scaled_sum.cuh"

__device__ const bool DEBUG = true;

namespace cgrps = cooperative_groups;

/** @brief Prints the desired matrix in row-column order.
 * @param T *matrix - pointer to the stored matrix
 * @param uint32 rows - number of rows in matrix
 * @param uint32 columns - number of columns
 * */
template <typename T>
__host__ __device__ void printMatrix(T *matrix, uint32_t rows, uint32_t cols)
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

/** @brief The rsLQR solver, the main function of the solver
 */
template <typename T>
__global__ void solve_Kernel(uint32_t nhorizon,
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
  copy2<float>(cost_step*nhorizon,1,d_Q_R,s_Q_R, dyn_step*nhorizon,1,d_A_B,s_A_B);
  copy2<float>((nstates+ninputs)*nhorizon,-1.0, d_q_r,s_q_r, nstates*nhorizon,-1.0, d_d,s_d);
  copy3<float>(states_sq*nhorizon*depth,1,d_F_lambda,s_F_lambda, states_sq*nhorizon*depth, 1,
  d_F_state,s_F_state, inp_states*nhorizon*depth, 1,d_F_input,s_F_input);
  block.sync(); // block or grid?

  // building the tree
  initializeBSTLevels(nhorizon, s_levels);
  block.sync();

  // should solveLeaf in parallel, each block per time step
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    solveLeaf<float>(s_levels, ind, nstates, ninputs, nhorizon,
                     s_Q_R, s_q_r, s_A_B, s_d,
                     s_F_lambda, s_F_state, s_F_input);

    block.sync();
    int cur_level = s_levels[ind];
    // copy result to ram need to copy the sol vector
    copy2<float>(nhorizon,1,s_d+(ind*nstates),d_d + (ind * nstates), ninputs+nstates,1,
     s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));

    // copy Q_R
    glass::copy(states_sq + inputs_sq, s_Q_R + (ind * cost_step),
                d_Q_R + (ind * cost_step));

    if (ind == 0)
    {
      // copy F_lambda&F_input only for ind==0
      copy2<float>(states_sq, 1, s_F_lambda, d_F_lambda, inp_states, 1, s_F_input, d_F_input);
    }

    else
    {
      // otherwise copy F-state prev
      int prev_level = s_levels[ind - 1];
      glass::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
                  d_F_state + ((prev_level * nhorizon + ind) * states_sq));

      if (ind < nhorizon - 1)
      {
        // copy cur_F_state + cur_F_input

        copy2<float>(states_sq,1,s_F_state+ + (states_sq * (cur_level * nhorizon + ind)),
        d_F_state+ (states_sq * (cur_level * nhorizon + ind)),
        inp_states,1, s_F_input+ (inp_states * (cur_level * nhorizon + ind)),
        d_F_input+ (inp_states * (cur_level * nhorizon + ind)));
      }
    }
  }
  grid.sync();

  // update the shared of soln vector
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>((nstates + ninputs) * nhorizon, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();

  // using only numleaves blocks (two timesteps per block)
  for (uint32_t level = 0; level < depth; level++)
  {

    // before processing make sure you copied all needed info from RAM
    // need to copy ind and ind+1 for all level - depth
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy F_lambda of ind and of ind+1
        copy3<float>(states_sq,1,d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        s_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        states_sq,1,d_F_state + (states_sq * (cur_level * nhorizon + ind)),
        s_F_state + (states_sq * (cur_level * nhorizon + ind)),
        inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
        s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        // copying ind+1
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
        s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)), 
        states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        s_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
      }
    }
    block.sync();

    uint32_t numleaves = pow(2.0, (depth - level - 1));
    uint32_t cur_depth = depth - level;

    // Calc Inner Products in parallel
    for (uint32_t b_ind = block_id; b_ind < numleaves; b_ind += grid_dim)
    {
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }
    block.sync(); // not needed
    // set to true to check factorInner
    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER calc_inner %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // Cholesky factorization
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
    }
    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER chol_solve %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // Solve with Cholesky factor for f
    uint32_t upper_levels = cur_depth - 1;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER solve_chol %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }
    // Solve updateShur
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / numleaves;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id;
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);

        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
      }
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER update_Schur %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // sanity check that we are done with the loop
    if (block_id == 0 && thread_id == 0)
    {
      printf("done with the loop!\n");
    }

    // after finishing with the level need to rewrite from shared to RAM only upperlevels of block
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t up_level = level + 1; up_level < depth; up_level++)
      {
        // copy F_lambda of ind and of ind+1
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (up_level * nhorizon + ind)),
        d_F_lambda + (states_sq * (up_level * nhorizon + ind)),
        states_sq, 1, s_F_state + (states_sq * (up_level * nhorizon + ind)),
        d_F_state + (states_sq * (up_level * nhorizon + ind)),
        inp_states, 1, s_F_input + (inp_states * (up_level * nhorizon + ind)),
        d_F_input + (inp_states * (up_level * nhorizon + ind)));

        // // copying ind+1
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (up_level * nhorizon + next_ind)),
        d_F_lambda + (states_sq * (up_level * nhorizon + next_ind)),
        states_sq, 1, s_F_state + (states_sq * (up_level * nhorizon + next_ind)),
        d_F_state + (states_sq * (up_level * nhorizon + next_ind)),
        inp_states, 1, s_F_input + (inp_states * (up_level * nhorizon + next_ind)),
        d_F_input + (inp_states * (up_level * nhorizon + next_ind)));
      }
      block.sync();
    }
  }

  //check RAM
if (DEBUG)
  {
    if (block_id == 0 && thread_id == 0)
    {
      printf("CHECKING D DATA AFTER Matrix Loop");
      for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
      {
        if (ind % nhorizon == 0)
        {
          printf("\nLEVEL %d\n", ind / nhorizon);
        }
        printf("\nF_lambda #%d: \n", ind);
        printMatrix(d_F_lambda + (ind * states_sq), nstates, nstates);

        printf("\nF_state #%d: \n", ind);
        printMatrix(d_F_state + (ind * states_sq), nstates, nstates);

        printf("\nF_input #%d: \n", ind);
        printMatrix(d_F_input + ind * inp_states, ninputs, nstates);
      }
      for (unsigned i = 0; i < nhorizon; i++)
      {
        printf("\nd%d: \n", i);
        printMatrix(d_d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
      }
    }
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING s_DATA AFTER matrix loop");
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
  }

  // solve for solution vector using the cached factorization
  // needs to have the right F_lambdas
  for (uint32_t level = 0; level < depth; ++level)
  {
    uint32_t numleaves = pow(2.0, (depth - level - 1));

    // calculate inner products with rhs, with factors computed above in parallel

    // copy soln vector
    /*
    int32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      // copy soln vector
      glass::copy(nhorizon, d_d + (ind * nstates), s_d + (ind * nstates));
      glass::copy(ninputs + nstates, d_q_r + (ind * (ninputs + nstates)), s_q_r + (ind * (ninputs + nstates)));
      // copy ind+1
      glass::copy(nhorizon, d_d + ((ind + 1) * nstates), s_d + ((ind + 1) * nstates));
      glass::copy(ninputs + nstates, d_q_r + ((ind + 1) * (ninputs + nstates)), s_q_r + ((ind + 1) * (ninputs + nstates)));
    }
    block.sync();*/

    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }
    grid.sync();
    // Solve for separator variables with cached Cholesky decomposition
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      float *Sbar = s_F_lambda + (level * nhorizon + lin_ind + 1) * (nstates * nstates);
      float *zy = s_d + (lin_ind + 1) * nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    grid.sync();

    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t k = block_id; k < nhorizon; k += grid_dim)
    {
      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels
      updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                            calc_lambda, nstates, ninputs, nhorizon);
    }
    grid.sync();
    // copy all the things for next level
    /*    
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      glass::copy(nhorizon, s_d + (ind * nstates), d_d + (ind * nstates));
      glass::copy(ninputs + nstates, s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));
      block.sync();
      // copy ind+1
      glass::copy(nhorizon, s_d + ((ind+1) * nstates), d_d + ((ind + 1) * nstates));
      glass::copy(ninputs + nstates, s_q_r + ((ind+1) * (ninputs + nstates)), d_q_r + ((ind + 1) * (ninputs + nstates)));
    }*/
    block.sync();
  }
  block.sync();

    // move shared to ram of soln vector
      
    for (unsigned i = thread_id; i < (nstates + ninputs) * nhorizon; i += block_dim)
    {
      d_q_r[i] = s_q_r[i];
    }
    for (unsigned i = thread_id; i < nhorizon * nstates; i += block_dim)
    {
      d_d[i] = s_d[i];
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECK FINAL RESULTS\n");
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printMatrix(s_d + i * nstates, nstates, 1);
          printMatrix(s_q_r + (i * (ninputs + nstates)), nstates, 1);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), ninputs, 1);
        }
      }
    }
    block.sync();
  }



  //NEW COMPARE TUES 20
  #pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
#include "./help_functions/nested_dissect.cuh"
#include "./help_functions/print_debug.cuh"
// should we put into one header file?


__device__ const bool DEBUG = true;

namespace cgrps = cooperative_groups;


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

/** @brief The rsLQR solver, the main function of the solver
 */
template <typename T>
__global__ void solve_Kernel(uint32_t nhorizon,
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
  copy2<float>(cost_step*nhorizon,1,d_Q_R,s_Q_R, dyn_step*nhorizon,1,d_A_B,s_A_B);
  copy2<float>((nstates+ninputs)*nhorizon,-1.0, d_q_r,s_q_r, nstates*nhorizon,-1.0, d_d,s_d);
  copy3<float>(states_sq*nhorizon*depth,1,d_F_lambda,s_F_lambda, states_sq*nhorizon*depth, 1,
  d_F_state,s_F_state, inp_states*nhorizon*depth, 1,d_F_input,s_F_input);
  block.sync(); // block or grid?

  // building the tree
  initializeBSTLevels(nhorizon, s_levels);
  block.sync();

  block.sync();
  // should solveLeaf in parallel, each block per time step
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    solveLeaf<float>(s_levels, ind, nstates, ninputs, nhorizon,
                     s_Q_R, s_q_r, s_A_B, s_d,
                     s_F_lambda, s_F_state, s_F_input);

    block.sync();
    int cur_level = s_levels[ind];
    // copy result to ram need to copy the sol vector
    copy2<float>(nhorizon,1,s_d+(ind*nstates),d_d + (ind * nstates), ninputs+nstates,1,
     s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));

    // copy Q_R
    glass::copy(states_sq + inputs_sq, s_Q_R + (ind * cost_step),
                d_Q_R + (ind * cost_step));

    if (ind == 0)
    {
      // copy F_lambda&F_input only for ind==0
      copy2<float>(states_sq, 1, s_F_lambda, d_F_lambda, inp_states, 1, s_F_input, d_F_input);
    }

    else
    {
      // otherwise copy F-state prev
      int prev_level = s_levels[ind - 1];
      glass::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
                  d_F_state + ((prev_level * nhorizon + ind) * states_sq));

      if (ind < nhorizon - 1)
      {
        // copy cur_F_state + cur_F_input

        copy2<float>(states_sq,1,s_F_state+ + (states_sq * (cur_level * nhorizon + ind)),
        d_F_state+ (states_sq * (cur_level * nhorizon + ind)),
        inp_states,1, s_F_input+ (inp_states * (cur_level * nhorizon + ind)),
        d_F_input+ (inp_states * (cur_level * nhorizon + ind)));
      }
    }
  }
  grid.sync();

  // update the shared of soln vector
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>((nstates + ninputs) * nhorizon, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();
  
  // using only numleaves blocks (two timesteps per block)
  for (uint32_t level = 0; level < depth; level++)
  {

    // before processing make sure you copied all needed info from RAM
    // need to copy ind and ind+1 for all level - depth
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy F_lambda of ind and of ind+1
        copy3<float>(states_sq,1,d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        s_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        states_sq,1,d_F_state + (states_sq * (cur_level * nhorizon + ind)),
        s_F_state + (states_sq * (cur_level * nhorizon + ind)),
        inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
        s_F_input + (inp_states * (cur_level * nhorizon + ind)));
        block.sync();

        // copying ind+1
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
        s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)), 
        states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        s_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
        block.sync();
      }
    }
    block.sync();
  //check RAM
if (!DEBUG)
  {
    if (block_id == 0 && thread_id == 0)
    {
      printf("CHECKING D DATA AFTER Matrix Loop level : %d," , level);
      for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
      {
        if (ind % nhorizon == 0)
        {
          printf("\nLEVEL %d\n", ind / nhorizon);
        }
        printf("\nF_lambda #%d: \n", ind);
        printMatrix(d_F_lambda + (ind * states_sq), nstates, nstates);

        printf("\nF_state #%d: \n", ind);
        printMatrix(d_F_state + (ind * states_sq), nstates, nstates);

        printf("\nF_input #%d: \n", ind);
        printMatrix(d_F_input + ind * inp_states, ninputs, nstates);
      }
      for (unsigned i = 0; i < nhorizon; i++)
      {
        printf("\nd%d: \n", i);
        printMatrix(d_d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
      }
    }
  }
if(!DEBUG){
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING s_DATA AFTER matrix loop, level : %d," , level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
      }
    block.sync();
    uint32_t numleaves = pow(2.0, (depth - level - 1));
    uint32_t cur_depth = depth - level;

    // Calc Inner Products in parallel
    for (uint32_t b_ind = block_id; b_ind < numleaves; b_ind += grid_dim)
    {
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }
    block.sync(); // not needed
    // set to true to check factorInner
    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER calc_inner %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // Cholesky factorization
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER chol_solve %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // Solve with Cholesky factor for f
    uint32_t upper_levels = cur_depth - 1;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER solve_chol %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }
    
    // Solve updateShur
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / numleaves;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id;
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);

        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
      }
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECKING DATA AFTER update_Schur %d", level);
        for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
        {
          if (ind % nhorizon == 0)
          {
            printf("\nLEVEL %d\n", ind / nhorizon);
          }
          printf("\nF_lambda #%d: \n", ind);
          printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);

          printf("\nF_state #%d: \n", ind);
          printMatrix(s_F_state + (ind * states_sq), nstates, nstates);

          printf("\nF_input #%d: \n", ind);
          printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        }
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printf("\nd%d: \n", i);
          printMatrix(s_d + i * nstates, 1, nstates);

          printf("\nq%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        }
      }
    }

    // sanity check that we are done with the loop
    if (block_id == 0 && thread_id == 0)
    {
      printf("done with the loop!\n");
    }

    // after finishing with the level need to rewrite from shared to RAM only upperlevels of block
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t up_level = level; up_level < depth; up_level++)
      {
        // copy F_lambda of ind and of ind+1
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (up_level * nhorizon + ind)),
        d_F_lambda + (states_sq * (up_level * nhorizon + ind)),
        states_sq, 1, s_F_state + (states_sq * (up_level * nhorizon + ind)),
        d_F_state + (states_sq * (up_level * nhorizon + ind)),
        inp_states, 1, s_F_input + (inp_states * (up_level * nhorizon + ind)),
        d_F_input + (inp_states * (up_level * nhorizon + ind)));

        // // copying ind+1
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (up_level * nhorizon + next_ind)),
        d_F_lambda + (states_sq * (up_level * nhorizon + next_ind)),
        states_sq, 1, s_F_state + (states_sq * (up_level * nhorizon + next_ind)),
        d_F_state + (states_sq * (up_level * nhorizon + next_ind)),
        inp_states, 1, s_F_input + (inp_states * (up_level * nhorizon + next_ind)),
        d_F_input + (inp_states * (up_level * nhorizon + next_ind)));
      }
      block.sync();
    }
  }



  // solve for solution vector using the cached factorization
  // needs to have the right F_lambdas
  for (uint32_t level = 0; level < depth; ++level)
  {
    uint32_t numleaves = pow(2.0, (depth - level - 1));

    // calculate inner products with rhs, with factors computed above in parallel

    // copy soln vector
    /*
    int32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      // copy soln vector
      glass::copy(nhorizon, d_d + (ind * nstates), s_d + (ind * nstates));
      glass::copy(ninputs + nstates, d_q_r + (ind * (ninputs + nstates)), s_q_r + (ind * (ninputs + nstates)));
      // copy ind+1
      glass::copy(nhorizon, d_d + ((ind + 1) * nstates), s_d + ((ind + 1) * nstates));
      glass::copy(ninputs + nstates, d_q_r + ((ind + 1) * (ninputs + nstates)), s_q_r + ((ind + 1) * (ninputs + nstates)));
    }
    block.sync();*/

    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }
    grid.sync();
    // Solve for separator variables with cached Cholesky decomposition
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      float *Sbar = s_F_lambda + (level * nhorizon + lin_ind + 1) * (nstates * nstates);
      float *zy = s_d + (lin_ind + 1) * nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    grid.sync();

    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t k = block_id; k < nhorizon; k += grid_dim)
    {
      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels
      updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                            calc_lambda, nstates, ninputs, nhorizon);
    }
    grid.sync();
    // copy all the things for next level
    /*    
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      glass::copy(nhorizon, s_d + (ind * nstates), d_d + (ind * nstates));
      glass::copy(ninputs + nstates, s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));
      block.sync();
      // copy ind+1
      glass::copy(nhorizon, s_d + ((ind+1) * nstates), d_d + ((ind + 1) * nstates));
      glass::copy(ninputs + nstates, s_q_r + ((ind+1) * (ninputs + nstates)), d_q_r + ((ind + 1) * (ninputs + nstates)));
    }*/
    block.sync();
  }
  block.sync();

    // move shared to ram of soln vector
      
    for (unsigned i = thread_id; i < (nstates + ninputs) * nhorizon; i += block_dim)
    {
      d_q_r[i] = s_q_r[i];
    }
    for (unsigned i = thread_id; i < nhorizon * nstates; i += block_dim)
    {
      d_d[i] = s_d[i];
    }

    if (!DEBUG)
    {
      if (block_id == 0 && thread_id == 0)
      {
        printf("CHECK FINAL RESULTS\n");
        for (unsigned i = 0; i < nhorizon; i++)
        {
          printMatrix(s_d + i * nstates, nstates, 1);
          printMatrix(s_q_r + (i * (ninputs + nstates)), nstates, 1);
          printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), ninputs, 1);
        }
      }
    }
    block.sync();
  }


  if(block_id == 1 && thread_id == 0) {
          printf("\nstart d_F_state %p, s_F_state %p",(void*)d_F_state,(void*)s_F_state);
          printf("\npointers, cur d_F_state %p, s_F_state %p,",
         (void*)(d_F_state + (states_sq * (cur_level * nhorizon + ind))),
          (void*)(s_F_state + (states_sq * (cur_level * nhorizon + ind))));
        }
        
  if(block_id == 1 && thread_id == 0) {
          printf("\npointers, d_F_state %p, s_F_state %p," ,(void*)(d_F_state + (states_sq * (cur_level * nhorizon + next_ind))),
       (void*)(s_F_state + (states_sq * (cur_level * nhorizon + next_ind))));
        }
        block.sync();
        if (DEBUG)
        {
          if (block_id == 1 && thread_id == 0)
          {
            printf("CHECKING DATA AFTER SOLVE LEAF");
            printf("ind, %d , next_ind, %d", ind, next_ind);
            print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                             d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
          }
          block.sync();
        }








        //NEW COMPARE

         // check after copying to SHARED
      if (!DEBUG)
      {
        if (block_id == BLOCK && thread_id == THREAD)
        {

          printf("CHECKING DATA AFTER COPY TO SHARED, level %d", level);
          print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                           d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
        }
        block.sync();
      }
    
    // set to true to check factorInner
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER factorInner, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }

      // check after Chol_inPlace
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {
        printf("CHECKING DATA AFTER chol_inPlace, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }
// check after solve_chole
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER solve_chol, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }


    // check after update_schur
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER update_schur, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }


        // do we need sync?
    block.sync();
    // check after copy to RAM
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER copy to RAM, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }
  




  #pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
#include "./help_functions/print_debug.cuh"
#include "./help_functions/nested_dissect.cuh"

__device__ const bool DEBUG = true;
__device__ const int BLOCK = 0;
__device__ const bool THREAD = 0;

namespace cgrps = cooperative_groups;
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

/** @brief The rsLQR solver, the main function of the solver
 */
template <typename T>
__global__ void solve_Kernel(uint32_t nhorizon,
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

  // initial point checks
  if (!DEBUG)
    if (block_id == BLOCK && thread_id == THREAD)
    {
      // check Lambdas
      printf("Init Address of d_F_lambda: %lu\n", (unsigned long)(void *)d_F_lambda);
      printf("Init Address of s_F_lambda %lu\n", (unsigned long)(void *)s_F_lambda);
      // //check states
      // printf("Init Address of d_F_state: %p\n", (unsigned long)(void*)d_F_state);
      // printf(" Init Address of s_F_state: %p\n", (unsigned long)(void*)s_F_state);
      // //check input
      // printf("Init Address of d_F_lambda: %p\n", (unsigned long)(void*)d_F_input);
      // printf("Init Address of s_F_lambda %p\n", (unsigned long)(void*)s_F_input);
    }

  // move ram to shared
  copy2<float>(cost_step * nhorizon, 1, d_Q_R, s_Q_R, dyn_step * nhorizon, 1, d_A_B, s_A_B);
  copy2<float>((nstates + ninputs) * nhorizon, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
  copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
               d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);
  block.sync(); // block or grid?
  initializeBSTLevels(nhorizon, s_levels);
  block.sync();

  // should solveLeaf in parallel, each block per time step
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    solveLeaf<float>(s_levels, ind, nstates, ninputs, nhorizon,
                     s_Q_R, s_q_r, s_A_B, s_d,
                     s_F_lambda, s_F_state, s_F_input);

    block.sync();
    int cur_level = s_levels[ind];
    // copy result to ram need to copy the sol vector
    copy2<float>(nhorizon, 1, s_d + (ind * nstates), d_d + (ind * nstates), ninputs + nstates, 1,
                 s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));

    // copy Q_R
    glass::copy(states_sq + inputs_sq, s_Q_R + (ind * (states_sq + inputs_sq)),
                d_Q_R + (ind * (states_sq + inputs_sq)));

    if (ind == 0)
    {
      // copy F_lambda&F_input only for ind==0
      copy2<float>(states_sq, 1, s_F_lambda, d_F_lambda, inp_states, 1, s_F_input, d_F_input);
    }

    else
    {
      // otherwise copy F-state prev
      int prev_level = s_levels[ind - 1];
      glass::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
                  d_F_state + ((prev_level * nhorizon + ind) * states_sq));

      if (ind < nhorizon - 1)
      {
        // copy cur_F_state + cur_F_input
        copy2<float>(states_sq, 1, s_F_state + +(states_sq * (cur_level * nhorizon + ind)),
                     d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                     inp_states, 1, s_F_input + (inp_states * (cur_level * nhorizon + ind)),
                     d_F_input + (inp_states * (cur_level * nhorizon + ind)));
      }
    }
  }
  grid.sync();

  // update the shared ONLY of the soln vector (factors updated later in the loop)
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>((nstates + ninputs) * nhorizon, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();

  // set to true to check solveLeaf & copying
  if (!DEBUG)
  {
    if (block_id == BLOCK && thread_id == THREAD)
    {
      printf("CHECKING DATA AFTER SOLVE LEAF");
      printf("\nstart d_F_state %p, s_F_state %p", (void *)d_F_state, (void *)s_F_state);
      print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                       d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
    }
    block.sync();
  }

  // using only numleaves blocks (two timesteps per block)
  for (uint32_t level = 0; level < depth; level++)
  {
    // before processing make sure you copied all needed info from RAM
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        uint32_t next_ind = ind + 1;

        // checking pointers
        if (DEBUG)
          if (block_id == BLOCK && thread_id == THREAD)
          {
            //print the array
            for(uint32_t i=0;i <nhorizon;i++){ printf("\narray %d, %d ",i, s_tree_result[i]);}
            printf("\ncount %d, ind %d, level %d,\n", count, ind, cur_level);

            // check Lambdas
            printf("Address of d_F_lambda: %lu , s_F_lambda %lu\n",
                   ((unsigned long)(float *)(d_F_lambda + (states_sq * (cur_level * nhorizon + ind)))) % 10000,
                   ((unsigned long)((float *)s_F_lambda + (states_sq * (cur_level * nhorizon + ind)))) % 10000);
            printf("Address of d_F_lambda: %lu , s_F_lambda %lu\n",
                   ((unsigned long)(float *)(d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)))) % 10000,
                   ((unsigned long)(float *)(s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)))) % 10000);
            // //check states
            // printf("Address of d_F_state: %lu\n",
            // (unsigned long)(void*)d_F_state+ (states_sq * (cur_level * nhorizon + ind)));
            // printf("Address of s_F_state: %lu\n",
            // (unsigned long)(void*)s_F_state+ (states_sq * (cur_level * nhorizon + ind)));
            // //check input
            // printf("Address of d_F_lambda: %lu\n",
            // (unsigned long)(void*)d_F_input + (inp_states * (cur_level * nhorizon + ind)));
            // printf("Address of s_F_lambda %lu\n",
            // (unsigned long)(void*)s_F_input + (inp_states * (cur_level * nhorizon + ind)));
          }
        // copy F_lambda of ind and of ind+1
        // copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        //              s_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
        //              states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
        //              s_F_state + (states_sq * (cur_level * nhorizon + ind)),
        //              inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
        //              s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        glass::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                    s_F_lambda + (states_sq * (cur_level * nhorizon + ind)));
        glass::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        // copying ind+1
        // copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
        //              s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
        //              states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        //              s_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
        //              inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
        //              s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
// 
        glass::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                    s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)));
        glass::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + next_ind)));
        glass::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));

        block.sync();
        //     if (block_id == BLOCK && thread_id == THREAD)
        // {
        //   printf("count %d, ind %d, level %d,\n", count, ind, cur_level);
        // }
      }
    }

    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER COPY TO SHARED, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }
    uint32_t numleaves = pow(2.0, (depth - level - 1));
    uint32_t cur_depth = depth - level;

    // Calc Inner Products in parallel
    for (uint32_t b_ind = block_id; b_ind < numleaves; b_ind += grid_dim)
    {
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }
    // set to true to check factorInner
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER factorInner, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }

    // Cholesky factorization
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
    }
    // check after Chol_inPlace
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {
        printf("CHECKING DATA AFTER chol_inPlace, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }
    // Solve with Cholesky factor for f
    uint32_t upper_levels = cur_depth - 1;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
    }
    // check after solve_chole
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER solve_chol, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }

    // Solve updateShur
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / numleaves;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id;
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);

        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
      }
    }
    // check after update_schur
    if (DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER update_schur, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }


    count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    // after finishing with the level need to rewrite from shared to RAM only upperlevels of block
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];

      for (uint32_t copy_level = level; copy_level < depth; copy_level++)
      {
        uint32_t next_ind = ind + 1;

        // checking pointers
        if (!DEBUG)
          if (block_id == BLOCK && thread_id == THREAD)
          {
            for(uint32_t i=0;i <nhorizon;i++){ printf("\nend array %d, %d ",i, s_tree_result[i]);}

            printf("\ncount %d, ind %d, level %d,\n", count, ind, copy_level);

            // check Lambdas
            printf("Address of d_F_lambda: %lu , s_F_lambda %lu\n",
                   ((unsigned long)(float *)(d_F_lambda + (states_sq * (copy_level * nhorizon + ind)))) % 10000,
                   ((unsigned long)((float *)(s_F_lambda + (states_sq * (copy_level * nhorizon + ind)))) % 10000));
            printf("Address of d_F_lambda: %lu , s_F_lambda %lu\n",
                   ((unsigned long)(float *)(d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)))) % 10000,
                   ((unsigned long)(float *)(s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)))) % 10000);
            // //check states
            // printf("Address of d_F_state: %lu\n",
            // (unsigned long)(void*)d_F_state+ (states_sq * (cur_level * nhorizon + ind)));
            // printf("Address of s_F_state: %lu\n",
            // (unsigned long)(void*)s_F_state+ (states_sq * (cur_level * nhorizon + ind)));
            // //check input
            // printf("Address of d_F_lambda: %lu\n",
            // (unsigned long)(void*)d_F_input + (inp_states * (cur_level * nhorizon + ind)));
            // printf("Address of s_F_lambda %lu\n",
            // (unsigned long)(void*)s_F_input + (inp_states * (cur_level * nhorizon + ind)));
          }

        // copy F_lambda of ind and of ind+1
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                  d_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                 states_sq, 1, s_F_state + (states_sq * (copy_level * nhorizon + ind)),
                 d_F_state + (states_sq * (copy_level * nhorizon + ind)),
                 inp_states, 1, s_F_input + (inp_states * (copy_level * nhorizon + ind)),
                  d_F_input + (inp_states * ( copy_level * nhorizon + ind)));

        // glass::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
        //             d_F_lambda + (states_sq * (copy_level * nhorizon + ind)));
        // glass::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + ind)),
        //             d_F_state + (states_sq * (copy_level * nhorizon + ind)));
        // glass::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + ind)),
        //             d_F_input + (inp_states * (copy_level * nhorizon + ind)));

        // copying ind+1
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                  d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                 states_sq, 1, s_F_state + (states_sq * (copy_level * nhorizon +next_ind)),
                 d_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
                 inp_states, 1, s_F_input + (inp_states * (copy_level * nhorizon + next_ind)),
                  d_F_input + (inp_states * (copy_level * nhorizon + next_ind)));

        // glass::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
        //             d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)));
        // glass::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
        //             d_F_state + (states_sq * (copy_level * nhorizon + next_ind)));
        // glass::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + next_ind)),
        //             d_F_input + (inp_states * (copy_level * nhorizon + next_ind)));
      }
    }
    // block.sync();

    // check after copy to RAM
    if (!DEBUG)
    {
      if (block_id == BLOCK && thread_id == THREAD)
      {

        printf("CHECKING DATA AFTER copy to RAM, level %d", level);
        print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                         d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
      }
      block.sync();
    }
        // sanity check that we are done with the loop
    if (block_id == BLOCK && thread_id == THREAD)
    {
      printf("done with the loop %d!\n", level);
    }
  }

  // CHECK RAM before the soln vector
  //  check after copy to RAM
  if (!DEBUG)
  {
    if (block_id == BLOCK && thread_id == THREAD)
    {

      // printf("CHECKING SHARED of final loop");
      // print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth,
      //           nstates, ninputs);

      // printf("CHECKING RAM of final loop");
      // print_KKT(d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth,
      //           nstates, ninputs);
      print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                       d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
    }
    block.sync();
  }

  // do we need sync?
  // grid.sync();

  // solve for solution vector using the cached factorization
  // HAVEN"T DONE COPYiNG TO FROM RAM FOR SOLN!!
  for (uint32_t level = 0; level < depth; ++level)
  {
    // print what we have in the beginning
    uint32_t numleaves = pow(2.0, (depth - level - 1));

    // calculate inner products with rhs, with factors computed above
    // in parallel
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }
    grid.sync();
    // Solve for separator variables with cached Cholesky decomposition
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      float *Sbar = s_F_lambda + (level * nhorizon + lin_ind + 1) * (nstates * nstates);
      float *zy = s_d + (lin_ind + 1) * nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    grid.sync();
    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t k = block_id; k < nhorizon; k += grid_dim)
    {
      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels
      updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                            calc_lambda, nstates, ninputs, nhorizon);
    }
    grid.sync();
  }
  block.sync();

  // move shared to ram of soln vector
  for (unsigned i = thread_id; i < (nstates + ninputs) * nhorizon; i += block_dim)
  {
    d_q_r[i] = s_q_r[i];
  }
  for (unsigned i = thread_id; i < nhorizon * nstates; i += block_dim)
  {
    d_d[i] = s_d[i];
  }

  if (!DEBUG)
  {
    if (block_id == BLOCK && thread_id == THREAD)
    {
      printf("CHECK FINAL RESULTS\n");
      for (unsigned i = 0; i < nhorizon; i++)
      {
        printMatrix(s_d + i * nstates, nstates, 1);
        printMatrix(s_q_r + (i * (ninputs + nstates)), nstates, 1);
        printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), ninputs, 1);
      }
    }
  }
  block.sync();
}


April 8
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
#include "./help_functions/print_debug.cuh"
#include "./help_functions/nested_dissect.cuh"
#include "./help_functions/tree_functs.cuh"
__device__ const bool DEBUG = true;
__device__ const int BLOCK = 0;
__device__ const bool THREAD = 0;
__device__ int error_flag = 0;

namespace cgrps = cooperative_groups;

/** @brief The rsLQR solver, the main function of the solver
 */
template <typename T>
__global__ void solve_Kernel_t(uint32_t nhorizon,
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
  int *s_levels = (int *)s_F_input + (depth * inp_states * nhorizon);
  int *s_tree_result = (int *)(s_levels + nhorizon);

  // move ram to shared
  copy2<float>(cost_step * nhorizon, 1, d_Q_R, s_Q_R, dyn_step * nhorizon, 1, d_A_B, s_A_B);
  copy2<float>((nstates + ninputs) * nhorizon, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
  copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
               d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);
  block.sync(); // block or grid?
  initializeBSTLevels(nhorizon, s_levels);
  block.sync();

  // should solveLeaf in parallel, each block per time step
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    solveLeaf<float>(s_levels, ind, nstates, ninputs, nhorizon,
                     s_Q_R, s_q_r, s_A_B, s_d,
                     s_F_lambda, s_F_state, s_F_input);

    block.sync();
    int cur_level = s_levels[ind];
    // copy result to ram need to copy the sol vector
    copy2<float>(nhorizon, 1, s_d + (ind * nstates), d_d + (ind * nstates), ninputs + nstates, 1,
                 s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));

    // copy Q_R
    glass::copy(states_sq + inputs_sq, s_Q_R + (ind * (states_sq + inputs_sq)),
                d_Q_R + (ind * (states_sq + inputs_sq)));

    if (ind == 0)
    {
      // copy F_lambda&F_input only for ind==0
      copy2<float>(states_sq, 1, s_F_lambda, d_F_lambda, inp_states, 1, s_F_input, d_F_input);
    }

    else
    {
      // otherwise copy F-state prev
      int prev_level = s_levels[ind - 1];
      glass::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
                  d_F_state + ((prev_level * nhorizon + ind) * states_sq));

      if (ind < nhorizon - 1)
      {
        // copy cur_F_state + cur_F_input
        copy2<float>(inp_states, 1, s_F_input + (inp_states * (cur_level * nhorizon + ind)),
                     d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                     states_sq, 1, s_F_state + +(states_sq * (cur_level * nhorizon + ind)),
                     d_F_state + (states_sq * (cur_level * nhorizon + ind)));
      }
    }
  }

  grid.sync();
  // update the shared ONLY of the soln vector (factors updated in main loop)
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>((nstates + ninputs) * nhorizon, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();


  for (uint32_t level = 0; level < depth; level++)
  {
    // before processing make sure you copied all needed info from RAM
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    uint32_t numleaves = pow(2.0, (depth - level - 1));
    uint32_t cur_depth = depth - level;
    // NEW
    uint32_t upper_levels = cur_depth - 1;
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / numleaves;

    // COPY
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      if (ind == -1)
      {
        atomicExch(&error_flag, 1);
        printf("BUG!\n");
        return;
      }
      // copy the timestep at level 0 for factor_inner
      copy3<float>(states_sq, 1, d_F_lambda + (states_sq * ind),
                   s_F_lambda + (states_sq * ind),
                   states_sq, 1, d_F_state + (states_sq * ind),
                   s_F_state + (states_sq * ind),
                   inp_states, 1, d_F_input + (inp_states * ind),
                   s_F_input + (inp_states * ind));
      block.sync();
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy ind and ind+ 1 for current level and upper levels
        if (level != 0)
        {
          copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                       s_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                       states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                       s_F_state + (states_sq * (cur_level * nhorizon + ind)),
                       inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                       s_F_input + (inp_states * (cur_level * nhorizon + ind)));
        }
        // copy next_ind
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                     s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                     states_sq, 1, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                     s_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                     inp_states, 1, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                     s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
      }
      block.sync();
    }

    //   // // copy for update Shur (double copying here a LOT,try to find a better way)
    if (level != depth - 1 && level != 0)
    {
      for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
      {
        uint32_t num_copies = nhorizon / count;
        uint32_t ind = s_tree_result[b_id];
        for (uint32_t k = b_id * num_copies; k < num_copies; k++)
        {
          for (uint32_t cur_level = level; cur_level < depth; cur_level++)
          {
            ind = k + nhorizon * level;
            copy3<float>(states_sq, 1, d_F_lambda + (states_sq * ind),
                         s_F_lambda + (states_sq * ind),
                         states_sq, 1, d_F_state + (states_sq * ind),
                         s_F_state + (states_sq * ind),
                         inp_states, 1, d_F_input + (inp_states * ind),
                         s_F_input + (inp_states * ind));
          }
        }
      }
    }

    // Calc Inner Products in parallel
    for (uint32_t b_ind = block_id; b_ind < numleaves; b_ind += grid_dim)
    {
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }
    // Cholesky factorization
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim)
    {
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
      block.sync();
    }

    // Solve with Cholesky factor for f
    // uint32_t upper_levels = cur_depth - 1;
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
      block.sync();
    }

    // // Solve updateShur

    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id;
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        int f = (index + 1) + nhorizon * upper_level;
        int F = k + nhorizon * level;
        int g = k + nhorizon * upper_level;

        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
        
        block.sync();
        // copy whatever was touched in Update Shur
        glass::copy(states_sq, s_F_lambda + (states_sq * f),
                    d_F_lambda + (states_sq * f));
        glass::copy(states_sq, s_F_state + (states_sq * f),
                    d_F_state + (states_sq * f));
        glass::copy(inp_states, s_F_input + (inp_states * f),
                    d_F_input + (inp_states * f));
        // // copy F
        glass::copy(states_sq, s_F_lambda + (states_sq * F),
                    d_F_lambda + (states_sq * F));
        glass::copy(states_sq, s_F_state + (states_sq * F),
                    d_F_state + (states_sq * F));
        glass::copy(inp_states, s_F_input + (inp_states * F),
                    d_F_input + (inp_states * F));
        // copy g
        glass::copy(states_sq, s_F_lambda + (states_sq * g),
                    d_F_lambda + (states_sq * g));
        glass::copy(states_sq, s_F_state + (states_sq * g),
                    d_F_state + (states_sq * g));
        glass::copy(inp_states, s_F_input + (inp_states * g),
                    d_F_input + (inp_states * g));
        block.sync();
      }
      block.sync();
    }
    
    grid.sync(); //not sure if needed
    // after finishing with the level need to rewrite from shared to RAM only upperlevels of block
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      if (ind == -1)
      {
        atomicExch(&error_flag, 1);
        return;
      }
      // copy the timestep at level 0 for factor_inner
      copy3<float>(states_sq, 1, s_F_lambda + (states_sq * ind),
                     d_F_lambda + (states_sq * ind),
                     states_sq, 1, s_F_state + (states_sq * ind),
                     d_F_state + (states_sq * ind),
                     inp_states, 1, s_F_input + (inp_states * ind),
                     d_F_input + (inp_states *  ind));

      for (uint32_t copy_level = level; copy_level < depth; copy_level++)
      {
        if(level!=0) {
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                     d_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                     states_sq, 1, s_F_state + (states_sq * (copy_level * nhorizon + ind)),
                     d_F_state + (states_sq * (copy_level * nhorizon + ind)),
                     inp_states, 1, s_F_input + (inp_states * (copy_level * nhorizon + ind)),
                     d_F_input + (inp_states * (copy_level * nhorizon + ind)));
        }
        // copying ind+1
        uint32_t next_ind = ind + 1;
        copy3<float>(states_sq, 1, s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                     d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                     states_sq, 1, s_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
                     d_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
                     inp_states, 1, s_F_input + (inp_states * (copy_level * nhorizon + next_ind)),
                     d_F_input + (inp_states * (copy_level * nhorizon + next_ind)));
      }
      block.sync();
    }
    // sanity check that we are done with the loop
    if (block_id == BLOCK && thread_id == THREAD)
    {
      printf("done with the loop!\n");
      printf("count %d\n", count);
      for (int i = 0; i < nhorizon; i++)
      {
        printf("%d ", s_tree_result[i]);
      }
    }
    grid.sync(); // nt sure if needed
  }

  if (DEBUG)
  {
    if (block_id == BLOCK && thread_id == THREAD)
    {
      // printf("CHECKING DATA AFTER LOOP level\n");
      bool check1 = checkEquality(s_F_state + states_sq, d_F_state + states_sq, states_sq);
      bool check_all = checkEquality(s_F_state, d_F_state, states_sq * nhorizon * depth);
      bool check1_prl = checkEqual_prl(s_F_state + states_sq, d_F_state + states_sq, states_sq);
      bool check_all_prl = checkEqual_prl(s_F_state, d_F_state, states_sq * nhorizon * depth);

      printf("fstate: check1: %d, checkall: %d,check1_prl: %d, chekallprl:%d\n", check1 ? 1 : 0, check_all ? 1 : 0,
             check1_prl ? 1 : 0, check_all_prl ? 1 : 0);
      if (!check1 || !check_all)
        printf("ERROR!! FSTATE!");
      printf("CHECKING DATA AFTER loop");
      // print_KKT(d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon,
      //           depth, nstates, ninputs);
      print_ram_shared(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r,
                       d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
    }
    block.sync();
  }
}




NEW



    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t b_id = block_id; b_id < numleaves; b_id += grid_dim)
    {
      for (uint32_t i = 0; i < num_copies; i++)
      {
        uint32_t k = b_id+i;
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels

        // DEBUG - check shur
        // check what's involved here
        if (SAFE_MOOD && block_id == BLOCK && thread_id == THREAD)
        {
          uint32_t f_d = index + 1;
          uint32_t g_d = k;
          uint32_t F_mat = k + nhorizon * level;
          printf("Checking shur_sol b_id %d,k %d,  f_soln %d,g_soln %d, F_mat %d\n", b_id,k, f_d, g_d, F_mat);
          // print_step_matrix(F_mat,s_F_lambda,s_F_state,s_F_input,nstates,ninputs);
          // print_soln_step(f_d,s_d,s_q_r,nhorizon,nstates,ninputs);
          // print_soln_step(g_d,s_d,s_q_r,nhorizon,nstates,ninputs);
        }
        block.sync();
        updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                              calc_lambda, nstates, ninputs, nhorizon);

        if (!SAFE_MOOD && block_id == BLOCK && thread_id == THREAD)
        {
          uint32_t f_d = index + 1;
          uint32_t g_d = k;
          uint32_t F_mat = k + nhorizon * level;
          printf("Checking shur_sol,  f_soln %d,g_soln %d, F_mat %d\n", f_d, g_d, F_mat);
          // print_step_matrix(F_mat,s_F_lambda,s_F_state,s_F_input,nstates,ninputs);
          // print_soln_step(f_d,s_d,s_q_r,nhorizon,nstates,ninputs);
          // print_soln_step(g_d,s_d,s_q_r,nhorizon,nstates,ninputs);
        }
        block.sync();
      }
    }
    block.sync();
    // return;