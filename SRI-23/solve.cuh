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