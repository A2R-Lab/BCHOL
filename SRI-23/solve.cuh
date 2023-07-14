#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
//should we put into one header file?

#include "./help_functions/chol_InPlace.cuh"
#include "./help_functions/chol_SolveInPlace.cuh"
#include "./help_functions/diag_Matrix_set.cuh"
#include "./help_functions/set_const.cuh"
#include "./help_functions/dot_product.cuh"
#include "./help_functions/scaled_sum.cuh"
#include "./help_functions/print_matrix.cuh"
__device__ const bool DEBUG = true;

namespace cgrps = cooperative_groups;

/* @brief Solve all the equations for the lower-level diagonal blocks, by timestep.
 * * For and LQR problem, it calculates
 *
 * \f[
 * Q_k^{-1} A_k^T
 * Q_k^{-1} q_k
 * R_k^{-1} B_k^T
 * R_k^{-1} r_k
 * \f]
 * @param index - timestep
 * @param nstates - number of states
 * @param ninputs - number of inputs
 * @param nhorizon - total number of timesteps
 * @param T *s_Q_R - array of Q_R matrices of timestep index
 * @param T *s_q_r - array of q_r vectors of timestep index
 * @param T *s_A_B - array of A_B matrices of timestep index
 * @param T *s_d - array of d vector of timestep index
 * @param T *s_F_lambda - an array to hold the result of 
 * @param T *s_F_state - an array to hold the result of 
 * @param T *s_F_input - an array to hold the result of 
 */

template <typename T> 
__device__
void solveLeaf(int *s_levels,
               uint32_t index,
               uint32_t nstates,
               uint32_t ninputs,
               uint32_t nhorizon,
               T *s_Q_R,
               T *s_q_r,
               T *s_A_B,
               T *s_d,
               T *s_F_lambda,
               T *s_F_state,
               T *s_F_input
              ) {

  //const for KKT matrix
  const uint32_t states_sq = nstates*nstates;
  const uint32_t inputs_sq = ninputs*ninputs;
  const uint32_t inp_states = ninputs*nstates;
  const uint32_t cost_step = states_sq+inputs_sq;
  const uint32_t dyn_step = states_sq+inp_states;


  //setting up arrays for specific indexes
  int level = s_levels[index];
  T* q = &s_q_r[index*(ninputs+nstates)];
  T* r = &s_q_r[index*(ninputs+nstates)+nstates];
  T* A = &s_A_B[index*dyn_step];
  T* B = &s_A_B[index*dyn_step+states_sq];
  T* d = &s_d[index*nstates];
  //fact matrices
  T* F_lambda = &s_F_lambda[(index + nhorizon * level)*states_sq];
  T* F_state = &s_F_state[(index+nhorizon*level)*states_sq];
  T* F_input = &s_F_input[(index+nhorizon*level)*inp_states];
  T* zy_temp;
  
  if (index == 0) {
    T* Q = &s_Q_R[cost_step*index]; 
    T* R = &s_Q_R[cost_step*index+states_sq]; 
    
    //Solve the block system of equations.
    glass::copy<T>(nstates*nstates, -1.0, A, F_lambda, cgrps::this_thread_block());

    // dont need to coz we initialized with 0s set_const(nstates*nstates,0, s_F_state); 
    set_const<T>(nstates*nstates,0.0, s_F_state);
    glass::copy<T>(nstates*ninputs,1.0, B, F_input); //copy  B_0
    chol_InPlace<T>(ninputs,R); //maybe unnessasry
    cholSolve_InPlace<T>(R, F_input, false, ninputs, nstates); //Fu = R\B
    cholSolve_InPlace<T>(R, r, false, ninputs, 1);  //zu = R\zu

    //Solve the block system of eqn (!overwriting d and q_r vectors!)
    zy_temp = &s_F_state[nhorizon*states_sq];
    glass::copy<T>(nstates,1.0,d,zy_temp); 
    glass::copy<T>(nstates,1.0, q, d);
    glass::gemv<T>(nstates,nstates,-1.0, Q, zy_temp, -1.0, d);  // zy = - Q * zy - zx
    glass::copy<T>(nstates,-1.0,zy_temp,q);
    set_const<T>(nstates, 0.0, zy_temp); //initialize back to 0s

  } else {

    T* Q = &s_Q_R[index*cost_step]; 
    chol_InPlace<T>(nstates,Q, cgrps::this_thread_block());

    //Not the last timestep
    if(index<nhorizon -1) {
      T* R = &s_Q_R[index*cost_step+states_sq];
      chol_InPlace<T>(ninputs,R, cgrps::this_thread_block());
      cholSolve_InPlace<T>(R, r,false, ninputs,1); //zu = R \ zu 

      glass::copy<T>(nstates*nstates,A, F_state);
      cholSolve_InPlace<T>(Q, F_state, false, nstates, nstates);

      glass::copy<T>(ninputs*nstates,1.0, B, F_input);
      cholSolve_InPlace<T>(R, F_input, false, ninputs, nstates);  
    }

    //Only term at the last timestep
    cholSolve_InPlace<T>(Q, q, false, nstates, 1); // Q\-q
    int prev_level = s_levels[index-1];
    T* F_state_prev = s_F_state+(index+nhorizon*prev_level)*states_sq; //prev level  F_state
    diag_Matrix_set<T>(nstates, -1.0 , F_state_prev);
    cholSolve_InPlace<T>(Q, F_state_prev, false, nstates, nstates); //solve Q \ -I from previous time step
  }
}

template <typename T> 
__device__
void factorInnerProduct(T* s_A_B, 
                        T* fact_state, 
                        T* fact_input, 
                        T* fact_lambda, 
                        int index, 
                        int fact_level, 
                        uint32_t nstates, 
                        uint32_t ninputs, 
                        uint32_t nhorizon) {

  int dyn_step = nstates*nstates+nstates*ninputs;

  T* C1_state = s_A_B + (index*dyn_step); 
  T* C1_input = s_A_B + (index*dyn_step + nstates*nstates);

  uint32_t linear_index = index + nhorizon * fact_level;
  T* F1_state = fact_state + linear_index*(nstates*nstates);
  T* F1_input = fact_input + linear_index*(ninputs*nstates);

  linear_index = (index + 1) + nhorizon * fact_level;
  T* F2_state = fact_state + linear_index*(nstates*nstates);
  T *S = fact_lambda + linear_index*(nstates*nstates); //F2_lambda

  dot_product<T>(nstates, nstates, nstates, 1.0, C1_state, F1_state, -1.0, S, cgrps::this_thread_block()); // S = C1x'F1x
  dot_product<T>(nstates, ninputs, nstates, 1.0, C1_input, F1_input, 1.0, S, cgrps::this_thread_block());
  scaled_sum<T>(nstates, nstates, -1.0, F2_state, S, cgrps::this_thread_block()); // equivalent to -I'F2_state 
}

template <typename T> 
__device__
void factorInnerProduct_sol(T* s_A_B, 
                        T* s_q_r, 
                        T* s_d,
                        int index,
                        uint32_t nstates, 
                        uint32_t ninputs, 
                        uint32_t nhorizon) {

  int dyn_step_AB = nstates*nstates+nstates*ninputs;
  int dyn_step_qr = nstates+ninputs;

  // Matrix
  T* C1_state = s_A_B + (index*dyn_step_AB); 
  T* C1_input = s_A_B + (index*dyn_step_AB + nstates*nstates);

  // Vector
  T* F1_state = s_q_r + (index*dyn_step_qr);
  T* F1_input = s_q_r + (index*dyn_step_qr + nstates);
  
  index += 1;
  T* F2_state = s_q_r + (index*dyn_step_qr);
  T *S = s_d + (index*nstates);

  dot_product<T>(nstates, nstates, 1, 1.0, C1_state, F1_state, -1.0, S, cgrps::this_thread_block()); // S = C1x'F1x
  dot_product<T>(nstates, ninputs, 1, 1.0, C1_input, F1_input, 1.0, S, cgrps::this_thread_block());
  scaled_sum<T>(nstates, 1, -1.0, F2_state, S, cgrps::this_thread_block()); // equivalent to -I'F2_state 
}

template <typename T> 
__device__
void SolveCholeskyFactor(T* fact_state, 
                         T* fact_input, 
                         T* fact_lambda, 
                         int index, 
                         int level, 
                         int upper_level, 
                         int nstates, 
                         int ninputs, 
                         int nhorizon) {
  T *Sbar = fact_lambda + ((index + 1) + nhorizon * level)*nstates*nstates;
  T *f = fact_lambda + ((index + 1) + nhorizon * upper_level)*nstates*nstates;
  
  cholSolve_InPlace<T>(Sbar, f, 0, nstates, nstates);
}

__device__
int getIndexFromLevel(int nhorizon, int depth, int level, int i, int* levels){
  int num_nodes = pow(2, depth - level - 1);
  int leaf = i * num_nodes / nhorizon;
  int count = 0;
  for (int k = 0; k < nhorizon; k++) {
    if (levels[k] != level) {continue;}
    if (count == leaf) {
      return k;
    }
    count ++;
  }
  return -1;
}

__device__
bool shouldCalcLambda(int index, int i, int nhorizon, int* levels){
  int left_start = index - int(pow(2,levels[index])) + 1;
  int right_start = index + 1;
  bool is_start = i == left_start || i == right_start;
  return !is_start || i == 0;
}

template <typename T> 
__device__
void updateShur(T* fact_state, 
                T* fact_input, 
                T* fact_lambda, 
                int index, 
                int i, 
                int level, 
                int upper_level, 
                bool calc_lambda, 
                int nstates, 
                int ninputs, 
                int nhorizon
               ) {
  T* g_state;
  T* g_input;
  T* g_lambda;

  T* F_state;
  T* F_input;
  T* F_lambda;

  int linear_index = (index + 1) + nhorizon * upper_level;
  T* f = fact_lambda + (linear_index * nstates*nstates);

  linear_index = i + nhorizon * upper_level;
  g_state = fact_state + (linear_index * (nstates *nstates));
  g_input = fact_input + (linear_index * (nstates*ninputs));
  g_lambda = fact_lambda + (linear_index * nstates*nstates);


  linear_index = i + nhorizon * level;
  F_state = fact_state + linear_index*nstates*nstates;
  F_input = fact_input + linear_index*nstates*ninputs;
  F_lambda = fact_lambda + linear_index*nstates*nstates;

  if(calc_lambda) {
    glass::gemm<T,0>(nstates, nstates, nstates, -1.0, F_lambda, f, 1.0, g_lambda,cgrps::this_thread_block());
  }
  glass::gemm<T,0>(nstates, nstates, nstates, -1.0, F_state, f, 1.0, g_state,cgrps::this_thread_block());
  glass::gemm<T,0>(ninputs, nstates, nstates, -1.0, F_input, f, 1.0, g_input,cgrps::this_thread_block());
}

template <typename T> 
__device__
void updateShur_sol(T* fact_state, 
                    T* fact_input, 
                    T* fact_lambda,
                    T* s_q_r,
                    T* s_d, 
                    int index, 
                    int i, 
                    int level, 
                    bool calc_lambda, 
                    int nstates, 
                    int ninputs, 
                    int nhorizon
                  ) {
  T* g_state;
  T* g_input;
  T* g_lambda;

  T* F_state;
  T* F_input;
  T* F_lambda;

  // ndlqr_GetNdFactor(soln, index + 1, 0, &f_factor);
  // Matrix* f = &f_factor->lambda;
  T* f = s_d + ((index+1)*nstates);

  // ndlqr_GetNdFactor(soln, i, 0, &g);
  int dyn_step_qr = nstates + ninputs;
  g_state = s_q_r + (i*dyn_step_qr);
  g_input = s_q_r + (i*dyn_step_qr + nstates);
  g_lambda = s_d + (i*nstates);

  // from fact, looks at level
  int linear_index = i + nhorizon * level;
  F_state = fact_state + linear_index*nstates*nstates;
  F_input = fact_input + linear_index*nstates*ninputs;
  F_lambda = fact_lambda + linear_index*nstates*nstates;

  if(calc_lambda) {
    glass::gemm<T,0>(nstates, nstates, 1, -1.0, F_lambda, f, 1.0, g_lambda,cgrps::this_thread_block());
  }
  glass::gemm<T,0>(nstates, nstates, 1, -1.0, F_state, f, 1.0, g_state,cgrps::this_thread_block());
  glass::gemm<T,0>(ninputs, nstates, 1, -1.0, F_input, f, 1.0, g_input,cgrps::this_thread_block());
}



__device__
void initializeBSTLevels(int nhorizon, int* levels) {
  int depth = log2f(nhorizon);

  for (int i = 0; i < nhorizon/2; i++){
    levels[2*i] = 0;
    levels[2*i+1] = -1;
  }

  bool toggle = false;
  int previous_index = -1;
  for (int i = 1; i < depth; i++){
    for (int j = 0; j < nhorizon; j++){
      if (levels[j] != i-1) {continue;}
      if (toggle) {
        levels[(previous_index + j)/2] = i;
        toggle = false;
      } else {
        previous_index = j;
        toggle = true;
      }
    }
  }
}


template <typename T>
__global__
void solve_Kernel(uint32_t* d_info,
                  T *d_Q_R,
                  T *d_q_r,
                  T *d_A_B,
                  T *d_d,
                  T *d_F_lambda,
                  T *d_F_state,
                  T *d_F_input
                  ){   

  printf("Launched Kernel\n");
  const cgrps::thread_block block = cgrps::this_thread_block();	 
  const cgrps::grid_group grid = cgrps::this_grid();
  const uint32_t block_id = blockIdx.x;
  const uint32_t block_dim = blockDim.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t grid_dim = gridDim.x;
  const uint32_t nhorizon = d_info[0];
  const uint32_t ninputs = d_info[1];
  const uint32_t nstates = d_info[2];
  const uint32_t depth = d_info[3];
  const uint32_t states_sq = nstates*nstates;
  const uint32_t inputs_sq = ninputs*ninputs;
  const uint32_t inp_states = ninputs*nstates;
  const uint32_t cost_step = states_sq+inputs_sq;
  const uint32_t dyn_step = states_sq+inp_states;

  printf("sth wrong here?\n");

  //move everything to shared memory
  extern __shared__ T s_temp[];
  T *s_Q_R = s_temp;
  T *s_q_r = s_Q_R + (cost_step)*nhorizon;
  T *s_A_B = s_q_r + (ninputs+nstates)*nhorizon;
  T *s_d = s_A_B + (dyn_step)*nhorizon;
  T *s_F_lambda = s_d + nstates*nhorizon;
  T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
  T *s_F_input = s_F_state + (states_sq *nhorizon* depth);
  int *s_levels = (int*)(s_F_input + depth*inp_states*nhorizon);

  //move ram to shared
  for(unsigned i = thread_id; i < (states_sq+inputs_sq)*nhorizon; i += block_dim){
    s_Q_R[i] = d_Q_R[i];
  }

  for(unsigned i = thread_id; i < (nstates+ninputs)*nhorizon; i +=block_dim) {
    s_q_r[i] = d_q_r[i];
  }

  for(unsigned i = thread_id; i < (states_sq+inp_states)*nhorizon; i +=block_dim) {
    s_A_B[i] = d_A_B[i]; 
  }
  
  for(unsigned i = thread_id; i < nhorizon*nstates; i += block_dim) {
    s_d[i] = d_d[i];
  }

  // initialize
  block.sync();

  if(thread_id == 0){
    initializeBSTLevels(nhorizon, s_levels);
  }

  //negate q_r,d vectors (using threads)
  for (uint32_t ind = thread_id; ind < (ninputs+nstates)*nhorizon; ind+=block_dim){
    s_q_r[ind] *= -1;
  }  
  
  for (uint32_t ind = thread_id; ind < (nstates)*nhorizon; ind+=block_dim){
    s_d[ind] *= -1;
  }

  //sync threads
  block.sync();

  //should solveLeaf in parallel
  for (uint32_t ind = block_id; ind < nhorizon; ind+=grid_dim) {
    solveLeaf<T>(s_levels, ind, nstates, ninputs, nhorizon,
                    s_Q_R, s_q_r, s_A_B,s_d, 
                    s_F_lambda, s_F_state, s_F_input);
  }

  //for some reason doesn't work when I call here  grid or block.sync()
  grid.sync();
  //block.sync();
  printf("done with solveLeaf\n");

  //Solve factorization - can do in parallel
  for(uint32_t level = 0; level < depth; level++) { //change to level < depth later
    uint32_t numleaves = pow(2.0, (depth-level-1));

    //Calc Inner Products
    uint32_t cur_depth = depth - level;
    uint32_t num_products = numleaves * cur_depth;

    //in parallel block or thread?
    for(uint32_t ind= block_id; ind < num_products; ind +=grid_dim) {
      uint32_t leaf = ind / cur_depth;
      uint32_t upper_level = level + (ind % cur_depth);
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      factorInnerProduct<T>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, upper_level, nstates, ninputs, nhorizon);
    }

    grid.sync();
    printf("done with innerproducts level %d\n",level);

    //Cholesky factorization
    for (uint32_t leaf = block_id; leaf < numleaves; leaf += grid_dim) {
      uint32_t index = pow(2.0, level) *(2*leaf+1)-1;
      uint32_t lin_ind = index + nhorizon*level;
      T* S = s_F_lambda+(states_sq*(lin_ind+1));
      chol_InPlace<T>(nstates, S, cgrps::this_thread_block());
    }

    grid.sync();
    printf("done with chol_fact level %d\n",level);

    //Solve with Cholesky factor for f
    uint32_t upper_levels = cur_depth-1;   
    uint32_t num_solves = numleaves*upper_levels;
    for (uint32_t i = block_id; i < num_solves; i+=grid_dim) {
      uint32_t leaf = i / upper_levels;
      uint32_t upper_level = level + 1 + (i % upper_levels);
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      SolveCholeskyFactor<T>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level, 
                                 nstates, ninputs, nhorizon);
    }

    grid.sync();
    printf("done with solve_chol level %d\n",level);

    //Shur compliments
    uint32_t num_factors = nhorizon * upper_levels;
    for (uint32_t i = thread_id ; i < num_factors; i+=block_dim) {
      int k = i/upper_levels;
      uint32_t upper_level = level+1+( i %upper_levels);

      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
      updateShur<T>(s_F_state, s_F_input, s_F_lambda, index, k,  level, upper_level, 
                        calc_lambda, nstates,  ninputs, nhorizon);
    }
    block.sync();
        
    printf("done with update_shur level %d\n",level);
  }


  //solve for solution vector using the cached factorization
  for (uint32_t level = 0; level < depth; ++level) {
    uint32_t numleaves = pow(2.0, (depth-level-1) );

    //calculate inner products with rhs, with factors computed above
    //in parallel
    for(uint32_t leaf = thread_id; leaf < numleaves; leaf+=block_dim) {
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct_sol(s_A_B, s_q_r, s_d, lin_ind, nstates, ninputs, nhorizon);
    }
    block.sync();
    printf("done with factor_inner_sol level %d\n",level);

    //Solve for separator variables with cached Cholesky decomposition
    for(uint32_t leaf = thread_id; leaf < numleaves; leaf +=block_dim) {
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      T* Sbar = s_F_lambda + (level*nhorizon+lin_ind+1)*(nstates*nstates);
      T* zy = s_d + (lin_ind+1)*nstates;
      // Sbar \ z = zbar
      cholSolve_InPlace(Sbar, zy, false, nstates, 1);
    }
    block.sync();
    printf("done with chol_solve_sol level %d\n",level);

    // Propagate information to solution vector
    //    y = y - F zbar
    for(uint32_t k = thread_id; k < nhorizon; k+=block_dim) {
      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k,nhorizon, s_levels); // nhorizon, s_levels
      updateShur_sol<T>(s_F_state,s_F_input,s_F_lambda, s_q_r, s_d, index, k , level,
                        calc_lambda, nstates,ninputs, nhorizon);
    }
    block.sync();
    printf("done with update_shur_sol level %d\n",level);
  }
  printf("done!\n");

  if(DEBUG) {
    if(block_id == 0 && thread_id == 0) {
      printf("CHECK FINAL RESULTS\n");
        for(unsigned i = 0; i < nhorizon; i++) {
          printMatrix(s_d+i*nstates,nstates,1);     
          printMatrix(s_q_r+(i*(ninputs+nstates)),nstates,1);
          printMatrix(s_q_r+(i*(ninputs+nstates)+nstates),ninputs,1);
        }
    }
  }
}
