#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../GLASS/glass.cuh"
#include "helpf.cuh"
__device__ const bool DEBUG = true;
__device__ const int BLOCK = 0;
__device__ const bool THREAD = 0;
__device__ int error_flag = 0;

namespace cgrps = cooperative_groups;
/** @brief The rsLQR solver, the main function of the solver
 */
template <typename T>
__global__ void solve_BCHOL(uint32_t nhorizon,
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
  const uint32_t soln_size = (nstates + ninputs) * nhorizon - ninputs;
  const uint32_t fstates_size = states_sq * nhorizon * depth;
  const uint32_t fcontrol_size = inp_states * nhorizon * depth;

  // maybe unneccesary
  const uint32_t Q_R_size = (cost_step) * (nhorizon - 1) + states_sq;
  const uint32_t q_r_size = (ninputs + nstates) * (nhorizon - 1) + nstates;

  // // initialize shared memory
  extern __shared__ T s_temp[];
  T *s_Q_R = s_temp;
  T *s_q_r = s_Q_R + Q_R_size;
  T *s_A_B = s_q_r + q_r_size;
  T *s_d = s_A_B + (dyn_step) * (nhorizon - 1);
  T *s_F_lambda = s_d + nstates * nhorizon;
  T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
  T *s_F_input = s_F_state + (states_sq * nhorizon * depth);
  int *s_levels = (int *)s_F_input + (depth * inp_states * nhorizon);
  int *s_tree_result = (int *)(s_levels + nhorizon);

  // Initialize d_F_lambdas to 0s
  set_const(fstates_size, 0.0f, d_F_lambda);
  set_const(fstates_size, 0.0f, d_F_state);
  set_const(fcontrol_size, 0.0f, d_F_input);

  // move ram to shared
  copy2<float>(Q_R_size, 1, d_Q_R, s_Q_R, dyn_step * (nhorizon - 1), 1, d_A_B, s_A_B);
  copy2<float>(q_r_size, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
  copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
               d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);
  initializeBSTLevels(nhorizon, s_levels); // helps to build the tree
  block.sync();



  // Make sure Q_R are non-zeros across diagonal
  for (int i = block_id; i < nhorizon; i++)
  {
    add_epsln(nstates, s_Q_R + i * cost_step);
  }
  block.sync();


  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("Check init\n"); // fixed!
    printf("B 0 %f", s_A_B[states_sq]);
    printf("\n B_4 %f",s_A_B[states_sq+4]);
    for (int i = 0; i < nhorizon - 1; i++)
    {
      printf("A %d\n", i);
      printMatrix(s_A_B + (i * (dyn_step)), nstates, nstates);
      printf("B %d\n", i);
      printMatrix(s_A_B + states_sq + (i * dyn_step), ninputs, nstates);
    }

    for (int i = 0; i < nhorizon; i++)
    {
      printf("Q %d\n", i);
      printMatrix(s_Q_R + (i * cost_step), nstates, nstates);
      if (i != nhorizon - 1)
      {
        printf("R %d\n", i);
        printMatrix(s_Q_R + states_sq + (i * cost_step), ninputs, ninputs);
      }
    }

    print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
  }
  block.sync();

  // GOOD TIL HERE FOR MPC INTEGRATION
  //  should solveLeaf in parallel, each block per time step
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
    glass_yana::copy(states_sq + inputs_sq, s_Q_R + (ind * (states_sq + inputs_sq)),
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
      glass_yana::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
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
  block.sync();

  block.sync();
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished solve_leaf\n");
    print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth, nstates, ninputs);
  }
  // block.sync()

  // update the shared ONLY of the soln vector (factors updated in main loop)
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>(q_r_size, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();

  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished solve_leaf and COPY\n");
    print_KKT(d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
  }

  for (uint32_t level = 0; level < depth; level++)
  {
    if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
    {
      printf("started big loop %d \n", level);
    }
    // before processing make sure you copied all needed info from RAM
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    uint32_t L = pow(2.0, (depth - level - 1)); // num of leaves
    uint32_t cur_depth = depth - level;
    uint32_t upper_levels = cur_depth - 1;
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / L;

    // COPY
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy ind and ind+ 1 for current level and upper levels
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                         s_F_lambda + (states_sq * (cur_level * nhorizon + ind)));
        glass_yana::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                         s_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass_yana::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                         s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        // copy next_ind
        uint32_t next_ind = ind + 1;
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                         s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)));
        glass_yana::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                         s_F_state + (states_sq * (cur_level * nhorizon + next_ind)));
        glass_yana::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                         s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
      }
    }

    // copy for update Shur (double copying here a LOT,try to find a better way)
    if (level != depth - 1 && level != 0)
    {
      for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
      {
        uint32_t num_copies = nhorizon / count;
        uint32_t ind = s_tree_result[b_id];
        for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
        {
          for (uint32_t cur_level = level; cur_level < depth; cur_level++)
          {

            ind = k + nhorizon * cur_level;
            glass_yana::copy(states_sq, d_F_lambda + (states_sq * ind),
                             s_F_lambda + (states_sq * ind));
            glass_yana::copy(states_sq, d_F_state + (states_sq * ind),
                             s_F_state + (states_sq * ind));
            glass_yana::copy(inp_states, d_F_input + (inp_states * ind),
                             s_F_input + (inp_states * ind));
          }
        }
      }
    }

    // Calc Inner Products Bbar and bbar (to solve for y)
    // here we change only F_lambda of ind+1 of fact_level
    for (uint32_t b_ind = block_id; b_ind < L; b_ind += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started factorinner loop \n");
        print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
      }
      // can be independent loop with threads!
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }

    // Cholesky factorization of Bbar/bbar
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started cholinplace loop \n");
      }
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
      block.sync();
    }
    if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
    {
      printf("After  chol_fact loop, level %d \n", level);
      print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth,
                nstates, ninputs);
    }
    block.sync();

    // Solve with Cholesky factor for y
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      // can add parallelization within threads here
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
        {
          printf("started cholfactor loop \n");
          print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
        }
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
    }
    if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
    {
      printf("After  chol_solve loop, level %d \n", level);
      print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth,
                nstates, ninputs);
    }
    block.sync();

    // update SHUR - update x and z compliments
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started shur loop \n");
      }
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id; //why is here 4?
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        if(!DEBUG&&block_id==BLOCK&&thread_id==THREAD){
          printf("level %d,i %d, calc_lambda %d\n",level,i,calc_lambda );
        }
        int g = k + nhorizon * upper_level;
        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
        // copy g
        block.sync();
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * g),
                         d_F_lambda + (states_sq * g));
        glass_yana::copy(states_sq, s_F_state + (states_sq * g),
                         d_F_state + (states_sq * g));
        glass_yana::copy(inp_states, s_F_input + (inp_states * g),
                         d_F_input + (inp_states * g));
      }
    }

    if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
    {
      printf("After  shur loop, level %d \n", level);
      print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth,
                nstates, ninputs);
    }
    block.sync();

    // after finishing with the level need to rewrite from shared to RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t copy_level = level; copy_level < depth; copy_level++)
      {
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                         d_F_lambda + (states_sq * (copy_level * nhorizon + ind)));
        glass_yana::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + ind)),
                         d_F_state + (states_sq * (copy_level * nhorizon + ind)));
        glass_yana::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + ind)),
                         d_F_input + (inp_states * (copy_level * nhorizon + ind)));

        // copying ind+1
        uint32_t next_ind = ind + 1;
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                         d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)));
        glass_yana::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
                         d_F_state + (states_sq * (copy_level * nhorizon + next_ind)));
        glass_yana::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + next_ind)),
                         d_F_input + (inp_states * (copy_level * nhorizon + next_ind)));
      }
      block.sync(); // not needed
    }
    grid.sync();
  }
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished fact big loop\n");
    print_KKT(s_F_lambda,s_F_state, s_F_input, s_d,s_q_r,nhorizon,depth,nstates,ninputs);
  }
  block.sync();


  // SOLN VECTOR LOOP
  for (uint32_t level = 0; level < depth; ++level)
  {

    uint32_t L = pow(2.0, (depth - level - 1));
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    uint32_t num_copies = nhorizon / count;

    // COPY from RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
      {
        uint32_t ind = k + level * nhorizon;
        glass_yana::copy((nstates + ninputs), d_q_r + k * (nstates + ninputs), s_q_r + k * (nstates + ninputs));
        glass_yana::copy(nstates, d_d + k * nstates, s_d + k * nstates);

        // copy the F_state
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * ind),
                         s_F_lambda + (states_sq * ind));
        glass_yana::copy(states_sq, d_F_state + (states_sq * ind),
                         s_F_state + (states_sq * ind));
        glass_yana::copy(inp_states, d_F_input + (inp_states * ind),
                         s_F_input + (inp_states * ind));
      }
    }

    // calculate inner products with rhs, with factors computed above
    // in parallel
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }

    // Solve for separator variables with cached Cholesky decomposition
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      float *Sbar = s_F_lambda + (level * nhorizon + lin_ind + 1) * states_sq;
      float *zy = s_d + (lin_ind + 1) * nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      for (uint32_t t_id = 0; t_id < num_copies; t_id++)
      {
        uint32_t k = b_id * num_copies + t_id;
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels
        if(!DEBUG&&block_id==BLOCK&&thread_id==THREAD){
          printf("L %d soln level %d,k %d,i %d, calc_lambda %d\n",L,level,k,index,calc_lambda );
        }
        updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                              calc_lambda, nstates, ninputs, nhorizon);
      }
    }
    block.sync();

    // COPY back to RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      // copy for update Shur
      uint32_t num_copies = nhorizon / count;
      for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
      {
        uint32_t ind = k + level * nhorizon;
        // copy the soln
        glass_yana::copy((nstates + ninputs), s_q_r + k * (nstates + ninputs), d_q_r + k * (nstates + ninputs));
        glass_yana::copy((nstates), s_d + k * nstates, d_d + k * nstates);
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * ind),
                         d_F_lambda + (states_sq * ind));
        glass_yana::copy(states_sq, s_F_state + (states_sq * ind),
                         d_F_state + (states_sq * ind));
        glass_yana::copy(inp_states, s_F_input + (inp_states * ind),
                         d_F_input + (inp_states * ind));
      }
    }
    grid.sync();
  }
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {

    printf("print HI soln vector\n");
    print_soln(d_d, d_q_r, nhorizon, nstates, ninputs);
  }
}

// VERSION FOR MPC INTEGRATION
/** @brief The rsLQR solver, the main function of the solver, here takes d_q_r and also d_c
 * doesn't solve in palce for vector solution
 */
template <typename T>
__global__ void solve_BCHOL_safe(uint32_t nhorizon,
                                 uint32_t ninputs,
                                 uint32_t nstates,
                                 T *d_Q_R,
                                 T *d_g,
                                 T *d_A_B,
                                 T *d_c,
                                 T *d_F_lambda,
                                 T *d_F_state,
                                 T *d_F_input,
                                 T *d_q_r,
                                 T *d_d)

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
  const uint32_t soln_size = (nstates + ninputs) * nhorizon - ninputs;
  const uint32_t fstates_size = states_sq * nhorizon * depth;
  const uint32_t fcontrol_size = inp_states * nhorizon * depth;

  // maybe unneccesary
  const uint32_t Q_R_size = (cost_step) * (nhorizon - 1) + states_sq;
  const uint32_t q_r_size = (ninputs + nstates) * (nhorizon - 1) + nstates;

  // // initialize shared memory
  extern __shared__ T s_temp[];
  T *s_Q_R = s_temp;
  T *s_q_r = s_Q_R + Q_R_size;
  T *s_A_B = s_q_r + q_r_size;
  T *s_d = s_A_B + (dyn_step) * (nhorizon - 1);
  T *s_F_lambda = s_d + nstates * nhorizon;
  T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
  T *s_F_input = s_F_state + (states_sq * nhorizon * depth);
  int *s_levels = (int *)s_F_input + (depth * inp_states * nhorizon);
  int *s_tree_result = (int *)(s_levels + nhorizon);

  // Initialize d_F_lambdas to 0s
  set_const(fstates_size, 0.0f, d_F_lambda);
  set_const(fstates_size, 0.0f, d_F_state);
  set_const(fcontrol_size, 0.0f, d_F_input);
  block.sync();

  // copy d and q_r for having the original g and c unchanged
  copy2<float>(q_r_size, 1.0, d_g, d_q_r, nstates * nhorizon, 1.0, d_d, d_g);
  block.sync();
  // move ram to shared
  copy2<float>(Q_R_size, 1, d_Q_R, s_Q_R, dyn_step * (nhorizon - 1), 1, d_A_B, s_A_B);
  copy2<float>(q_r_size, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
  copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
               d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);
  initializeBSTLevels(nhorizon, s_levels); // helps to build the tree
  block.sync();

  // Make sure Q_R are non-zeros across diagonal
  for (int i = block_id; i < nhorizon; i++)
  {
    add_epsln(nstates, s_Q_R + i * cost_step);
  }
  block.sync();

  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("Check init\n"); // fixed!
    for (int i = 0; i < nhorizon - 1; i++)
    {
      printf("A %d\n", i);
      printMatrix(s_A_B + (i * (dyn_step)), nstates, nstates);
      printf("B %d\n", i);
      printMatrix(s_A_B + states_sq + (i * dyn_step), nstates, ninputs);
    }
    for (int i = 0; i < nhorizon; i++)
    {
      printf("Q %d\n", i);
      printMatrix(s_Q_R + (i * cost_step), nstates, nstates);
      if (i != nhorizon - 1)
      {
        printf("R %d\n", i);
        printMatrix(s_Q_R + states_sq + (i * cost_step), ninputs, ninputs);
      }
    }
    print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
  }
  block.sync();

  // GOOD TIL HERE FOR MPC INTEGRATION
  //  should solveLeaf in parallel, each block per time step
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
    glass_yana::copy(states_sq + inputs_sq, s_Q_R + (ind * (states_sq + inputs_sq)),
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
      glass_yana::copy(states_sq, s_F_state + ((prev_level * nhorizon + ind) * states_sq),
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
  block.sync();

  block.sync();
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished solve_leaf\n");
    print_KKT(s_F_lambda, s_F_state, s_F_input, s_d, s_q_r, nhorizon, depth, nstates, ninputs);
  }
  // block.sync()

  // update the shared ONLY of the soln vector (factors updated in main loop)
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    copy2<float>(q_r_size, 1, d_q_r, s_q_r, nstates * nhorizon, 1, d_d, s_d);
  }
  grid.sync();

  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished solve_leaf and COPY\n");
    print_KKT(d_F_lambda, d_F_state, d_F_input, d_d, d_q_r, nhorizon, depth, nstates, ninputs);
  }

  for (uint32_t level = 0; level < depth; level++)
  {
    if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
    {
      printf("started big loop %d \n", level);
    }
    // before processing make sure you copied all needed info from RAM
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    uint32_t L = pow(2.0, (depth - level - 1)); // num of leaves
    uint32_t cur_depth = depth - level;
    uint32_t upper_levels = cur_depth - 1;
    uint32_t num_factors = nhorizon * upper_levels;
    uint32_t num_perblock = num_factors / L;

    // COPY
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy ind and ind+ 1 for current level and upper levels
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                         s_F_lambda + (states_sq * (cur_level * nhorizon + ind)));
        glass_yana::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                         s_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass_yana::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                         s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        // copy next_ind
        uint32_t next_ind = ind + 1;
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                         s_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)));
        glass_yana::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                         s_F_state + (states_sq * (cur_level * nhorizon + next_ind)));
        glass_yana::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                         s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
      }
    }

    // copy for update Shur (double copying here a LOT,try to find a better way)
    if (level != depth - 1 && level != 0)
    {
      for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
      {
        uint32_t num_copies = nhorizon / count;
        uint32_t ind = s_tree_result[b_id];
        for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
        {
          for (uint32_t cur_level = level; cur_level < depth; cur_level++)
          {

            ind = k + nhorizon * cur_level;
            glass_yana::copy(states_sq, d_F_lambda + (states_sq * ind),
                             s_F_lambda + (states_sq * ind));
            glass_yana::copy(states_sq, d_F_state + (states_sq * ind),
                             s_F_state + (states_sq * ind));
            glass_yana::copy(inp_states, d_F_input + (inp_states * ind),
                             s_F_input + (inp_states * ind));
          }
        }
      }
    }

    // Calc Inner Products Bbar and bbar (to solve for y)
    for (uint32_t b_ind = block_id; b_ind < L; b_ind += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started factorinner loop \n");
        print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
      }
      // can be independent loop with threads!
      for (uint32_t t_ind = 0; t_ind < cur_depth; t_ind += 1)
      {
        uint32_t ind = b_ind * cur_depth + t_ind;
        uint32_t leaf = ind / cur_depth;
        uint32_t upper_level = level + (ind % cur_depth);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
      }
    }

    // Cholesky factorization of Bbar/bbar
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started cholinplace loop \n");
      }
      // HERE"S THE PROBLEM!
      uint32_t index = pow(2.0, level) * (2 * leaf + 1) - 1;
      uint32_t lin_ind = index + nhorizon * level;
      float *S = s_F_lambda + (states_sq * (lin_ind + 1));
      chol_InPlace<float>(nstates, S, cgrps::this_thread_block());
      block.sync();
    }

    // Solve with Cholesky factor for y
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      // can add parallelization within threads here
      for (uint32_t t_id = 0; t_id < upper_levels; t_id += 1)
      {
        if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
        {
          printf("started cholfactor loop \n");
          print_soln(s_d, s_q_r, nhorizon, nstates, ninputs);
        }
        uint32_t i = b_id * upper_levels + t_id;
        uint32_t leaf = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
        SolveCholeskyFactor<float>(s_F_lambda, lin_ind, level, upper_level,
                                   nstates, ninputs, nhorizon);
      }
    }

    // update SHUR - update x and z compliments
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
      {
        printf("started shur loop \n");
      }
      for (uint32_t t_id = 0; t_id < num_perblock; t_id += 1)
      {
        int i = (b_id * 4) + t_id;
        int k = i / upper_levels;
        uint32_t upper_level = level + 1 + (i % upper_levels);
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
        int g = k + nhorizon * upper_level;
        updateShur<float>(s_F_state, s_F_input, s_F_lambda, index, k, level, upper_level,
                          calc_lambda, nstates, ninputs, nhorizon);
        // copy g
        block.sync();
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * g),
                         d_F_lambda + (states_sq * g));
        glass_yana::copy(states_sq, s_F_state + (states_sq * g),
                         d_F_state + (states_sq * g));
        glass_yana::copy(inp_states, s_F_input + (inp_states * g),
                         d_F_input + (inp_states * g));
      }
    }

    // after finishing with the level need to rewrite from shared to RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t copy_level = level; copy_level < depth; copy_level++)
      {
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + ind)),
                         d_F_lambda + (states_sq * (copy_level * nhorizon + ind)));
        glass_yana::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + ind)),
                         d_F_state + (states_sq * (copy_level * nhorizon + ind)));
        glass_yana::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + ind)),
                         d_F_input + (inp_states * (copy_level * nhorizon + ind)));

        // copying ind+1
        uint32_t next_ind = ind + 1;
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)),
                         d_F_lambda + (states_sq * (copy_level * nhorizon + next_ind)));
        glass_yana::copy(states_sq, s_F_state + (states_sq * (copy_level * nhorizon + next_ind)),
                         d_F_state + (states_sq * (copy_level * nhorizon + next_ind)));
        glass_yana::copy(inp_states, s_F_input + (inp_states * (copy_level * nhorizon + next_ind)),
                         d_F_input + (inp_states * (copy_level * nhorizon + next_ind)));
      }
      block.sync(); // not needed
    }
    grid.sync();
  }
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {
    printf("finished fact big loop\n");
  }

  // SOLN VECTOR LOOP
  for (uint32_t level = 0; level < depth; ++level)
  {

    uint32_t L = pow(2.0, (depth - level - 1));
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    uint32_t num_copies = nhorizon / count;

    // COPY from RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
      {
        uint32_t ind = k + level * nhorizon;
        glass_yana::copy((nstates + ninputs), d_q_r + k * (nstates + ninputs), s_q_r + k * (nstates + ninputs));
        glass_yana::copy(nstates, d_d + k * nstates, s_d + k * nstates);

        // copy the F_state
        glass_yana::copy(states_sq, d_F_lambda + (states_sq * ind),
                         s_F_lambda + (states_sq * ind));
        glass_yana::copy(states_sq, d_F_state + (states_sq * ind),
                         s_F_state + (states_sq * ind));
        glass_yana::copy(inp_states, d_F_input + (inp_states * ind),
                         s_F_input + (inp_states * ind));
      }
    }

    // calculate inner products with rhs, with factors computed above
    // in parallel
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }

    // Solve for separator variables with cached Cholesky decomposition
    for (uint32_t leaf = block_id; leaf < L; leaf += grid_dim)
    {
      uint32_t lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1;
      float *Sbar = s_F_lambda + (level * nhorizon + lin_ind + 1) * states_sq;
      float *zy = s_d + (lin_ind + 1) * nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    // Propagate information to solution vector
    //    y = y - F zbar
    for (uint32_t b_id = block_id; b_id < L; b_id += grid_dim)
    {
      for (uint32_t i = 0; i < num_copies; i++)
      {
        uint32_t k = b_id * num_copies + i;
        int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
        bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels); // nhorizon, s_levels
        updateShur_sol<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, k, level,
                              calc_lambda, nstates, ninputs, nhorizon);
      }
    }
    block.sync();

    // COPY back to RAM
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      // copy for update Shur
      uint32_t num_copies = nhorizon / count;
      for (uint32_t k = b_id * num_copies; k < b_id * num_copies + num_copies; k++)
      {
        uint32_t ind = k + level * nhorizon;
        // copy the soln
        glass_yana::copy((nstates + ninputs), s_q_r + k * (nstates + ninputs), d_q_r + k * (nstates + ninputs));
        glass_yana::copy((nstates), s_d + k * nstates, d_d + k * nstates);
        glass_yana::copy(states_sq, s_F_lambda + (states_sq * ind),
                         d_F_lambda + (states_sq * ind));
        glass_yana::copy(states_sq, s_F_state + (states_sq * ind),
                         d_F_state + (states_sq * ind));
        glass_yana::copy(inp_states, s_F_input + (inp_states * ind),
                         d_F_input + (inp_states * ind));
      }
    }
    grid.sync();
  }
  if (!DEBUG && block_id == BLOCK && thread_id == THREAD)
  {

    printf("print HI soln vector\n");
    print_soln(d_d, d_q_r, nhorizon, nstates, ninputs);
  }
}