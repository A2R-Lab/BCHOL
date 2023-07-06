#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
//should we put into one header file?

#include "chol_InPlace.cuh"
#include "chol_SolveInPlace.cuh"
#include "./help_functions/diag_Matrix_set.cuh"
#include "./help_functions/set_const.cuh"
__device__ const bool DEBUG = true;

namespace cgrps = cooperative_groups;

/* @brief Solve all the equations for the lower-level diagonal blocks , by timestep.
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
  void printMatrix(T* matrix, uint32_t cols, uint32_t rows) {
  for (unsigned i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%.3f  ",matrix[i*cols+j]);
    }
    printf("\n");
  }
}

template <typename T> 
  __device__
  void solveLeaf(uint32_t level,
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
                 //T *s_c??
                ) {

  
  //const for KKT matrix
  const uint32_t states_sq = nstates*nstates;
  const uint32_t inputs_sq = ninputs*ninputs;
  const uint32_t inp_states = ninputs*nstates;
  const uint32_t cost_step = states_sq+inputs_sq;
  const uint32_t dyn_step = states_sq+inp_states;


  //setting up arrays for specific indexes
  float* q = &s_q_r[index*(ninputs+nstates)];
  float* r = &s_q_r[index*(ninputs+nstates)+nstates];
  float* A = &s_A_B[index*dyn_step];
  float* B = &s_A_B[index*dyn_step+states_sq];
  float* d = &s_d[index*nstates];
  //fact matrices
  float* F_lambda = &s_F_lambda[(index + nhorizon * level)*states_sq];
  float* F_state = &s_F_state[(index+nhorizon*level)*states_sq];
  float* F_input = &s_F_input[(index+nhorizon*level)*inp_states];
  //
  float* zy_temp = F_lambda+nstates*nstates;
  set_const<float>(nstates, 0.0, zy_temp); 
  
  if (index == 0) {
    float* Q = &s_Q_R[cost_step*index]; 
    float* R = &s_Q_R[cost_step*index+states_sq]; 
    
    //Solve the block system of equations.

    glass::copy<float>(nstates*nstates, -1.0, A, F_lambda, cgrps::this_thread_block());

    // dont need ti coz we initialized with 0s set_const(nstates*nstates,0, s_F_state); 

    glass::copy<float>(nstates*ninputs,1.0, B, F_input); //copy  B_0
    chol_InPlace<float>(ninputs, R); 
    cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates); //Fu = R\B
    cholSolve_InPlace<float>(R, r, false, ninputs, 1);  //zu = R\zu

    //Solve the block system of eqn (!overwriting d and q_r vectors!)
    glass::copy<float>(nstates,1.0,d,zy_temp); 
    glass::copy<float>(nstates,1.0, q, d);
    glass::gemv<float>(nstates,nstates,-1.0, Q, zy_temp, -1.0, d);  // zy = - Q * zy - zx
    glass::copy<float>(nstates,1.0,zy_temp,q);
    glass::scal<float>(nstates,-1.0, q,cgrps::this_thread_block());

    set_const<float>(nstates, 0.0, zy_temp); //initialize back to 0s

  } else {

    float* Q = &s_Q_R[index*cost_step]; 
    chol_InPlace<float>(nstates,Q);

    //Not the last timestep
    if(index<nhorizon -1) {
      float* R = &s_Q_R[index*cost_step+states_sq];
      chol_InPlace<float>(ninputs,R);
      cholSolve_InPlace<float>(R, r,false, ninputs,1); //zu = R \ zu 

      glass::copy<float>(nstates*nstates,A, F_state);
      cholSolve_InPlace<float>(Q, F_state, false, nstates, nstates);

      glass::copy<float>(ninputs*nstates,1.0,B, F_input);
      cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates);  //DOUBLE CHECK!

      //Initialize with -Identity matrix the next timestep
      //diag_Matrix_set<float>(nstates, -1.0 , F_state+states_sq);

    }
    //Only the last timestep
    cholSolve_InPlace<float>(Q, q, false, nstates, 1);        
    cholSolve_InPlace<float>(Q, F_state, false, nstates, nstates); //solve Q \ -I from previous time step
  }
  //Initialize with -Identity matrix the next timestep
  diag_Matrix_set<float>(nstates, -1.0 , F_state+states_sq);
  if(DEBUG) {
    for(uint32_t ind = 0; ind < nhorizon * 3 ;  ind++) {
          printf("INSIDE SOLVELEAF %d", index);
          if(ind%nhorizon==0){ 
            printf("\nLEVEL %d\n", ind/nhorizon);
          } 
            printf("\nF_lambda[%d]\n", ind);
            printMatrix(s_F_lambda+(ind*states_sq),nstates,nstates);

            printf("\nF_state%d: \n", ind);
            printMatrix(s_F_state+(ind*states_sq),nstates,nstates);

            printf("\nF_input%d: \n", ind);
            printMatrix(s_F_input+ind*inp_states, nstates,ninputs);

        }
  }
}

template <typename T> 
__device__
void factorInnerProduct(T* s_A_B, T* fact_state, T* fact_input, T* fact_lambda, int index, int data_level, int fact_level, uint32_t nstates, uint32_t ninputs, uint32_t nhorizon) {
  float* C1_state;
  float* C1_input;

  float* F1_state;
  float* F1_input;
  float* F1_lambda;

  float* C2_state;
  float* C2_input;

  float* F2_state;
  float* F2_input;
  float* F2_lambda;

  int dyn_step = nstates*nstates+nstates*ninputs;

  //no need for tree conversion int linear_index = index + nhorizon * data_level;
  C1_state = s_A_B + (index*dyn_step); 
  C1_input = s_A_B + (index*dyn_step + nstates*nstates);

  uint32_t linear_index = index + nhorizon * fact_level;
  F1_state = fact_state + linear_index*(nstates*nstates);
  F1_input = fact_input + linear_index*(nstates*nstates);
  F1_lambda = fact_lambda + linear_index*(nstates*ninputs);

  C2_state = s_A_B + (index * dyn_step);
  C2_input = s_A_B + (index * dyn_step + nstates * nstates);

  linear_index = (index + 1) + nhorizon * fact_level;
  F2_state = fact_state + linear_index*(nstates*nstates);
  F2_input = fact_input + linear_index*(nstates*nstates);
  F2_lambda = fact_input + linear_index*(nstates*ninputs);
  
  float *S = F2_lambda;
  glass::gemm<float,0>(nstates, nstates, ninputs, 1.0, C1_state, F1_state, -1.0, S,cgrps::this_thread_block()); //S = C1x'F1x
  glass::gemm<float,0>(nstates, ninputs, nstates, 1.0, C1_input, F1_input, 1.0,S,cgrps::this_thread_block());
  glass::gemm<float,0>(nstates, nstates, nstates, 1.0, C2_state, F2_state, 1.0,S,cgrps::this_thread_block());
  glass::gemm<float,0>(nstates, ninputs, nstates, 1.0, C2_input, F2_input, 1.0,S,cgrps::this_thread_block());
}

template <typename T> 
  __device__
  void SolveCholeskyFactor(T* fact_state, T* fact_input, T* fact_lambda, int index, int level, int upper_level, int nstates, int ninput, int nhorizon) {
  float *Sbar = fact_lambda + (index + 1) + nhorizon * level;
  float *f = fact_lambda + (index + 1) + nhorizon * upper_level;

  cholSolve_InPlace<float>(Sbar, f, 0, nstates, ninput);
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
  void updateShur(T* fact_state, T* fact_input, T* fact_lambda, T* q_r, T* d, int index, int i, int level, int upper_level, bool calc_lambda, int nstates, int ninputs, int nhorizon) {
  float* f_factor_state;
  float* f_factor_input;
  float* f_factor_lambda;

  float* g_state;
  float* g_input;
  float* g_lambda;

  float* F_state;
  float* F_input;
  float* F_lambda;

  int linear_index = (index + 1) + nhorizon * upper_level;
  f_factor_state = q_r + (linear_index * (nstates + ninputs));
  f_factor_input = q_r + (linear_index * (nstates + ninputs) + nstates);
  f_factor_lambda = d + (linear_index * nstates);

  linear_index = i + nhorizon * upper_level;
  g_state = q_r + (linear_index * (nstates + ninputs));
  g_input = q_r + (linear_index * (nstates + ninputs) + nstates);
  g_lambda = d + (linear_index * nstates);

  linear_index = i + nhorizon * level;
  F_state = fact_state + linear_index;
  F_input = fact_input + linear_index;
  F_lambda = fact_lambda + linear_index;

  float* f = f_factor_lambda;
  if(calc_lambda) {
    glass::gemm<float,0>(nstates, nstates, 1, -1.0, F_lambda, f, 1.0, g_lambda,cgrps::this_thread_block());
  }
  glass::gemm<float,0>(nstates, nstates, 1, -1.0, F_state, f, 1.0, g_state,cgrps::this_thread_block());
  glass::gemm<float,0>(ninputs, nstates, 1, -1.0, F_input, f, 1.0, g_input,cgrps::this_thread_block());
}



__device__
  void initializeBSTLevels(int nhorizon, int* levels) {
  int depth = log2f (nhorizon);

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
  void solve_Kernel(uint32_t nhorizon,    
                    uint32_t ninputs,
                    uint32_t nstates,
                    T *d_Q_R,
                    T *d_q_r,
                    T *d_A_B,
                    T *d_d,
                    T *d_F_lambda,
                    T *d_F_state,
                    T *d_F_input
                   ){   

  if(DEBUG)
    printf("Launched Kernel\n");
  //Ask Emre about cgrps again!
  const cgrps::thread_block block = cgrps::this_thread_block();	 
  const cgrps::grid_group grid = cgrps::this_grid();
  const uint32_t block_id = blockIdx.x;
  const uint32_t block_dim = blockDim.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t grid_dim=gridDim.x;

  //const for KKT matrix
  const uint32_t states_sq = nstates*nstates;
  const uint32_t inputs_sq = ninputs*ninputs;
  const uint32_t inp_states = ninputs*nstates;
  const uint32_t cost_step = states_sq+inputs_sq;
  const uint32_t dyn_step = states_sq+inp_states;

  const uint32_t depth = log2f(nhorizon);


  //move everything to shared memory
  extern __shared__ T s_temp[];
  T *s_Q_R = s_temp;
  T *s_q_r = s_Q_R + (cost_step)*nhorizon;
  T *s_A_B = s_q_r + (ninputs+nstates)*nhorizon;
  T *s_d = s_A_B + (dyn_step)*nhorizon;
  T *s_F_lambda = s_d + nstates*nhorizon;
  T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
  T *s_F_input = s_F_state + (states_sq *nhorizon* depth);
  int *s_levels = (int *)(s_F_input + depth*inp_states*nhorizon);

  //move ram to shared
  for(unsigned i = thread_id; i < (states_sq+inputs_sq)*nhorizon ; i += block_dim){
    s_Q_R[i] = d_Q_R[i];
  }

  for (unsigned i = thread_id; i < (nstates+ninputs)*nhorizon; i +=block_dim) {
    s_q_r[i] = d_q_r[i];
  }

  for (unsigned i = thread_id; i < (states_sq+inp_states)*nhorizon; i +=block_dim) {
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
  //sync threads
  block.sync();
  for (uint32_t ind = thread_id; ind < (nstates)*nhorizon; ind+=block_dim){
    s_d[ind] *= -1;
  }

  //sync threads
  block.sync();

  //checking initialization - WORKS
  if(!DEBUG){
    if(thread_id==0 && block_id ==0) {

      printf("PRINTING ALL DATA AFTER INIT\n");
      for(unsigned i = 0; i < nhorizon; i++) {
        printf("\nQ%d: \n", i);
        printMatrix(s_Q_R+(i*cost_step),nstates,nstates);

        printf("\nR%d: \n", i);
        printMatrix(s_Q_R+(i*cost_step+states_sq),ninputs,ninputs);

        printf("\nq%d: \n", i);
        printMatrix(s_q_r+(i*(ninputs+nstates)),1,nstates);

        printf("\nr%d: \n", i);
        printMatrix(s_q_r+(i*(ninputs+nstates)+nstates),1,ninputs);

        printf("\nA%d: \n", i);
        printMatrix(s_A_B+(i*dyn_step),nstates,nstates);

        printf("\nB%d: \n", i);
        printMatrix(s_A_B+(i*dyn_step+states_sq),nstates,ninputs);

        printf("\nd%d: \n", i);
        printMatrix(s_d+i*nstates,1,nstates);        
      } 

      for(uint32_t ind = 0; ind < nhorizon * depth ;  ind++) {
          if(ind%nhorizon==0){ 
            printf("\nLEVEL %d\n", ind/nhorizon);
          } 
            printf("\nF_lambda[%d]\n", ind);
            printMatrix(s_F_lambda+(ind*states_sq),nstates,nstates);

            printf("\nF_state%d: \n", ind);
            printMatrix(s_F_state+(ind*states_sq),nstates,nstates);

            printf("\nF_input%d: \n", ind);
            printMatrix(s_F_input+ind*inp_states, nstates,ninputs);

        }
    }    

  }

  //should solveLeaf in parallel

  for (uint32_t ind = block_id; ind < nhorizon; ind+=grid_dim) {
    int level = s_levels[ind];
    solveLeaf<float>(level, ind, nstates, ninputs, nhorizon,
                    s_Q_R, s_q_r, s_A_B,s_d, 
                    s_F_lambda, s_F_state, s_F_input);
  }

  //for some reason doesn't work when I call here grid.sync()
  block.sync();

  if(DEBUG) {

    if(block_id == 0 && thread_id == 0) {
      printf("CHECKING DATA AFTER SOLVE_LEAF");
        for(unsigned i = 0; i < nhorizon; i++) { 
          printf("\nd%d: \n", i);
          printMatrix(s_d+i*nstates,1,nstates);      

          printf("\nq%d: \n", i);
          printMatrix(s_q_r+(i*(ninputs+nstates)),1,nstates);

          printf("\nr%d: \n", i);
          printMatrix(s_q_r+(i*(ninputs+nstates)+nstates),1,ninputs);

        }
    }

      for(uint32_t ind = 0; ind < nhorizon * depth ;  ind++) {
        if(ind%nhorizon==0){ 
          printf("\nLEVEL %d\n", ind/nhorizon);
        }
        printf("\nF_lambda[%d]\n", ind);
        printMatrix(s_F_lambda+(ind*states_sq),nstates,nstates);

        printf("\nF_state%d: \n", ind);
        printMatrix(s_F_state+(ind*states_sq),nstates,nstates);

        printf("\nF_input%d: \n", ind);
        printMatrix(s_F_input+ind*inp_states, nstates,ninputs);

      }
  }
  block.sync();

  //Solve factorization -can do in parallel?
  for(uint32_t level = 0; level < depth; ++level ) {
    uint32_t numleaves = pow(2.0, (depth-level-1) );

    //Calc Inner Profucts
    uint32_t cur_depth = depth - level;
    uint32_t num_products = numleaves * cur_depth;

    //in parallel block or thread?
    for(uint32_t ind= block_id; ind < num_products; ind +=grid_dim) {
      uint32_t leaf = ind / cur_depth;
      uint32_t upper_level = level + (ind % cur_depth);
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      factorInnerProduct<float>(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level, nstates, ninputs, nhorizon);
      if(!DEBUG){
        if(thread_id==0 && block_id==0) {
          printf("factor inner product, leaf: %d; upper level: %d; lin_ind: %d\n", leaf, upper_level, lin_ind);
          // for(unsigned i = 0; i < nhorizon; i++) {
          //   printf("\nQ%d: \n", i);
          //   printMatrix(s_Q_R+(i*cost_step),nstates,nstates);

          //   printf("\nR%d: \n", i);
          //   printMatrix(s_Q_R+(i*cost_step+states_sq),ninputs,ninputs);

          //   printf("\nq%d: \n", i);
          //   printMatrix(s_q_r+(i*(ninputs+nstates)),1,nstates);

          //   printf("\nr%d: \n", i);
          //   printMatrix(s_q_r+(i*(ninputs+nstates)+nstates),1,ninputs);

          //   printf("\nA%d: \n", i);
          //   printMatrix(s_A_B+(i*dyn_step),nstates,nstates);

          //   printf("\nB%d: \n", i);
          //   printMatrix(s_A_B+(i*dyn_step+states_sq),nstates,ninputs);

          //   printf("\nd%d: \n", i);
          //   printMatrix(s_d+i*nstates,1,nstates);        
          // } 

          for(uint32_t ind = 0; ind < nhorizon * depth ;  ind++) {
            if(ind%nhorizon==0){ 
              printf("\nLEVEL %d\n", ind/nhorizon);
            } 
            printf("\nF_lambda[%d]\n", ind);
            printMatrix(s_F_lambda+(ind*states_sq),nstates,nstates);

            printf("\nF_state%d: \n", ind);
            printMatrix(s_F_state+(ind*states_sq),nstates,nstates);

            printf("\nF_input%d: \n", ind);
            printMatrix(s_F_input+ind*inp_states,nstates,ninputs);
          }
        }    
      }
    }
    //in original code syncs here before proceeding
    grid.sync();

    //Cholesky factorization- XIAN
    for (uint32_t leaf= block_id; leaf < numleaves; leaf += grid_dim) {
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      float* S = s_F_lambda+lin_ind+1;
      chol_InPlace<float>(nstates, S);
    }
    //Solve with Cholesky factor for f
    uint32_t upper_levels = cur_depth-1;   
    uint32_t num_solves = numleaves*upper_levels;
    //check if in block or grid
    for (uint32_t i =thread_id; i < num_solves; i+=block_dim) {
      uint32_t leaf = i / upper_levels;
      uint32_t upper_level = level + 1 + (i % upper_levels);
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      SolveCholeskyFactor<float>(s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level, nstates, ninputs, nhorizon);
    }

    printf("bye\n");

    //Shur compliments
    uint32_t num_factors = nhorizon * upper_levels;
    //thread or block?
    for (uint32_t i = thread_id ; i < num_factors; i+=block_dim) {
      int k = i/upper_levels;
      uint32_t upper_level = level+1+( i %upper_levels);

      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
      updateShur<float>(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d,  index, k,  level, upper_level, 
                        calc_lambda, nstates,  ninputs, nhorizon);
    }
    block.sync();
  }
  //solve for solution vector using the cached factorization
  for (uint32_t level = 0; level < depth; ++level) {
    uint32_t numleaves = pow(2.0, (depth-level-1) );

    //calculate inner products with rhs, with factors computed above
    //in parallel
    for(uint32_t leaf = thread_id; leaf < numleaves; leaf+=block_dim) {
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
      // Calculate z = d - F'b1 - F2'b2
      factorInnerProduct(s_A_B, s_d, s_q_r, s_q_r+nstates, lin_ind, level, 0, nstates, ninputs, nhorizon);
    }
    block.sync();
    //Solve for separator variables with cached Cholesky decomposition
    for(uint32_t leaf = thread_id; leaf < numleaves; leaf +=block_dim) {
      uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;

      // Get the Sbar Matrix calculated above
      float* Sbar = s_F_lambda + (level*nhorizon+lin_ind+1);
      float* zy = s_d + lin_ind+1;
      // Solve (S - C1'F1 - C2'F2)^{-1} (d - F1'b1 - F2'b2) -> Sbar \ z = zbar
      //                 |                       |
      //    reuse Cholesky factorization   Inner product calculated above
      //getSfactorization; Chek if Needed
      cholSolve_InPlace(Sbar, zy,false, nstates, nstates);
    }
    block.sync();
    // Propagate information to solution vector
    //    y = y - F zbar
    for(uint32_t k = thread_id; k < nhorizon; k+=block_dim) {
      int index = getIndexFromLevel(nhorizon, depth, level, k, s_levels);
      bool calc_lambda = shouldCalcLambda(index, k, nhorizon, s_levels);
      updateShur<float>(s_F_state,s_F_input,s_F_lambda, s_q_r, s_d, index, k , level, 0,
                        calc_lambda, nstates,ninputs, nhorizon);
    }
    block.sync();
  }
}
