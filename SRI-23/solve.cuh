#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#includ <cmath>
#include <cooperative_groups.h>
#include "./GLASS/GLASS.cuh"
//should we put into one header file?

#include "lowerBackSub.cuh"
#include "choleskyDecomp_InPlace.cuh"
#include "./help_functions/diag_Matrix_set.cuh"
#include "./help_functions/set_const.cuh"


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
void solveLeaf(uint32_t index,
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
               

    float* zy_temp;
    set_const(nstates, 0.0, zy_temp); //checked
               
    if (index ==0) {
        float* Q = &Q_R[0]; //CHECK WITH BRIAN!
        float* R = &Q_R[nstates*nstates]; 
               
        //Solve the block system of equations.
        copy(nstates*nstates, s_A_B, s_F_lambda); 
        scal(nstates*nstates,-1.0, s_F_lambda); //-A'_0 ; 
       // dont need ti coz we initialized with 0s set_const(nstates*nstates,0, s_F_state); 
        copy(nstates*ninputs,s_A_B+[nstates*nstates], s_F_input); //copy  B_0
        chol_InPlace(ninputs, R); 
        cholSolve_InPlace(R, s_F_input, ninputs, nstates); //ASK XIAN! Fu = R\B
        cholSolve_InPlace(R, s_q_r+[nstates], ninputs, 1);  //zu = R\zu

        //Solve the block system of eqn (!overwriting d and q_r vectors!)
        copy(nstates,d,zy_temp); 
        copy(nstates,q_r, d);
        gemv(nstates,nstates,-1.0, Q, zy_temp, -1.0, d);  // zy = - Q * zy - zx
        copy(nstates,zy_temp,q_r);
        scal(nstates,-1.0, q_r); // zx = -zy
        chol_InPlace(nstates, Q); //not sure if that's ok to have it inplace

    //CHECKED until here for agrs - recheck lowbackSub with Xian and A,B transpose
      } else {
          
        int level = 0;
        float* Q = &Q_R[0]; 
        chol_InPlace(nstates,Q);
        
        //Not the last timestep
        if(k<nhorizon -1) {
            float* R = &Q_R[nstaes*nstates];
            chol_InPlace(ninputs,R);
            cholSolve_InPlace(R, q_r+[nstates], ninputs,1); //zu = R \ zu 

            copy(nstates*nstates,s_A_B, s_F_state);
            cholSolve_InPlace(Q, s_F_state, nstates, nstates);

            copy(A_B+[nstates*nstates],F_input);
            cholSolve_InPlace(R,F_input,ninput, nstates);  //DOUBLE CHECK!
                   
           //Initialize with -Identity matrix the next timestep
            diag_Matrix_set(nstates*nstates, -1 , s_F_state[(nstates*nstates)]);
        }

        //Only the last timestep
         cholSolve_InPlace(Q,q_r, nstates, 1);
         
        
        //int prev_level = ndlqr_GetIndexLevel(&solver->tree,k-1); ??
         // NOTE: This is -I on the state for explicit integration
          //For implicit integrators we'd use the A2, B2 partials wrt the next
          //state and control
        //update k
        
        cholSolve_InPlace(Q, s_F_state, nstates, nstates); //solve Q \ -I from previous time step
        }
  }

template <typename T> 
__device__
void factorInnerProduct(T* A_B, T* fact_state, T* fact_input, T* fact_lambda, int index, int data_level, int fact_level, int nstates, int ninput, int nhorizon) {
    double* C1_state;
    double* C1_input;

    double* F1_state;
    double* F1_input;
    double* F1_lambda;

    double* C2_state;
    double* C2_input;

    double* F2_state;
    double* F2_input;
    double* F2_lambda;

    //no need for tree conversion int linear_index = index + nhorizon * data_level;
    C1_state = A_B+(index * dyn_step); 
    C1_input = A_B+(index * dyn_step + nstates * nstates);

    linear_index = index + nhorizon * fact_level; //Not sure
    F1_state = fact_state+linear_index;
    F1_input = fact_input+linear_index;
    F1_lambda = fact_lambda+linear_index;

    linear_index = (index + 1) + nhorizon * data_level;
    C2_state = A_B+(lindex * dyn_step);
    C2_input = A_B+(iindex * dyn_step + nstates * nstates);

    linear_index = (index + 1) + nhorizon * fact_level;
    F2_state = fact_state+linear_index;
    F2_input = fact_input+linear_index;
    F2_lambda = fact_input+linear_index;

    double *S = F2_lambda;
    gemm(nstates, nstates, ninput, 1.0, C1_state, F1_state, -1.0, S); //S = C1x'F1x, why -1.0 and not 0.0?
    gemm(nstates, ninput, nstates, 1.0, C1_input, F1_input, 1.0, S);
    gemm(nstates, nstates, nstates, 1.0, C2_state, F2_state, 1.0, S);
    gemm(nstates, ninput, nstates, C2_input, F2_input, 1.0, S);
}

template <typename T> 
__device__
void SolveCholeskyFactor(T* fact_state, T* fact_input, T* fact_lambda, int index, int level, int upper_level, int nstates, int ninput, int nhorizon) {
    T *Sbar = fact_lambda + (index + 1) + nhorizon * level;
    T *f = fact_lambda + (index + 1) + nhorizon * upper_level;

    cholSolve_InPlace(T *Sbar, T *f, 0, nstates, ninput);
}

template <typename T> 
__device__
void updateShur(T* fact_state, T* fact_input, T* fact_lambda, T* q_r, T* d, int index, int i, int level, int upper_level, bool calc_lambda, int nstates, int ninput, int nhorizon) {
    double* f_factor_state;
    double* f_factor_input;
    double* f_factor_lambda;

    double* g_state;
    double* g_input;
    double* g_lambda;

    double* F_state;
    double* F_input;
    double* F_lambda;

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

    double* f = f_factor_lambda;
    if(calc_lambda) {
        glass::gemm(nstates, nstates, 1, -1.0, F_lambda, f, 1.0, g_lambda);
    }
    gemm(nstates, nstates, 1, -1.0, F_state, f, 1.0, g_state);
    gemm(ninput, nstates, 1, -1.0, F_input, f, 1.0, g_input);
}


template <typename T>
__global__
void solve(uint32_t nhorizon,    
           uint32_t ninputs,
           uint32_t nstates,
           //T *d_x0, // don't think we need it, its in d[0]
           T *d_Q_R,
           T *d_q_r,
           T *d_A_B,
           T *d_d
           /*
           ,T *d_F_lambda,
           T *d_F_state,
           T *d_F_input*/ )
//what is exit_tol?
{
    //Ask Emre about cgrps again!
    const cgrps::thread_block block = cgrps::this_thread_block();	 
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t grid_dim=gridDim.x;
  
    //const for KKT matrix
    const unit32_t states_sq = nstates*nstates;
    const unit32_t inputs_sq = ninputs*ninputs;
    const uint32_t inp_states = ninputs*nstates;
    const unit32_t cost_step = states_sq+inputs_sq;
    const unit32_t dyn_step = states_sq+inp_states;
  
    const uint32_t depth = log2(nhorizon);
    
           
    //move everything to shared memory
    extern __shared__ T s_temp[];
    T *s_Q_R = s_temp;
    T *s_q_r = s_Q_R + (cost_step)*nhorizon;
    T *s_A_B = s_q_r + (ninputs+nstates)*nhorizon;
    T *s_d = s_A_B + (dyn_step)*nhorizon;

    //can I just continue point to the shared memory?
    T *s_F_lambda = s_d + (nstates*nhorizon);
    T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
    T *s_F_input = s_F_state + (states_sq *nhorizon* depth);
           
    //initialize ALL F matrices to 0s
    set_const(states_sq*nhorizon*depth+states_sq*nhorizon*depth+inp_states*nhorizon*depth,
               0, s_F_lambda);
           
    //negate q_r,d vectors (using threads)
    for (uint32_t ind = thread_id; ind < (ninput+nstates)*nhorizon; ind+=block_dim){
               s_q_r[ind] *= -1;
    }
    //sync threads
    grid.sync()
      
    for (uint32_t ind = thread_id; ind < (nstates)*nhorizon; ind+=block_dim){
  
               s_d[ind] *= -1;
    }
    
    //sync threads
    grid.sync()
           
    //should solveLeaf in parallel
    for (uint32_t ind = block_id; ind < nhorizon; ind+=grid_dim) {
           int level = static_cast<int> (log2((1+ind) & -1 * (1 + ind)));
//maybe add level to args in solveleaf?
           solveleaf(ind, nstates, ninputs, nhorizon, s_Q_R+ind*(cost_step),
                     s_q_r+ind*(ninputs+nstates), s_A_B+ind*(dyn_step),
                     s_d+ind*nstates, s_F_lambda, s_F_state, s_F_input);
                     //s_F_lambda+ind*(states_sq), s_F_state+ind*(states_sq),s_F_input+ind*(inputs_sq) for F matrices use getIndexTree?
    }
    
    grid.sync();

    if(debug) {
      if(block_id == 0 && thread_id == 0) {
        for(uint32_t ind = 0; ind < nhorizon * depth ;  ind++) {
          printf("s_F_lambda[%d], %.2f\n", ind, s_F_lambda[i]);
          printf("s_F_state[%d], %.2f\n", ind, s_F_state[i]);
          printf("s_F_input[%d],%.2f\n", ind, s_F_input[i]);
        }
      }
    }
    grid.sync();
    
    //Solve factorization -can do in parallel?
    for(uint32_t level = 0; level < depth; ++level ) {
      uint32_t numleaves = pow(2.0, (depth-level-1) );

      //Calc Inner Profucts
      uint32_t cur_depth = depth - level;
      uint32_t num_products = numleaves * cur_depth;

      //in parallel block or thread?
      for(uint32_t ind= block_id; i < numproducts; ind +=grid_dim) {
       uint32_t leaf = ind/ cur_depth;
       uint32_t upper_level = level+ (i/cur_depth);
       uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
       factorInnerProduct(s_A_B, s_F_state, s_F_input, s_F_lambda, lin_ind, level, upper_level, states, ninput, nhorizon);
      }
      //in original code syncs here before proceeding
      grid.sync();
      
      //Cholesky factorization- XIAN
      for (uint32_t leaf= block_id; leaf < numleaves; leaf += grid_dim) {
        uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;
        float* S = F_lambda[lin_ind+1];
        //get Sbat matrix calculated above

      }
      //Solve with Cholesky factor for f
      uint32_t upper_levels = cur_depth-1;   
      uint32_t num_solves = numleaves*upper_levels;
      //check if in block or grid
      for (uint32_t i =thread_id; i < num_solves; i+=block_dim) {

      }

      //Shur compliments
      uint32_t num_factors = nhorizon *upper_levels;
      //thread or block?
      for (uint32_t i = thread_id ; i < num_factors; i+=block_dim) {
        uint32_t k = i/upper_levels;
        uint32_t upper_level = level+1+( i %upper_levels);

        uint32_t index ; //FIGURE OUT
        bool calc_lambda = shouldCalcLambda();
        updateShur(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d,  index,  i,  level, upper_level, 
                    calc_lambda, nstates,  ninput, nhorizon);
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
          factorInnerProduct(s_A_B, s_d, s_q_r, s_q_r+nstates, lin_ind, level, 0, nstates, ninput, nhorizon);
      }
      block.sync();
    //Solve for separator variables with cached Cholesky decomposition
      for(uint32_t leaf = thread_id; leaf < numleaves; leaf +=block_dim) {
          uint32_t lin_ind = pow(2.0, level) *(2*leaf+1)-1;

           // Get the Sbar Matrix calculated above
           float* Sbar = s_F_lambda + (level*nhorizon+index+1);
           float* zy = s_d + index+1;
        // Solve (S - C1'F1 - C2'F2)^{-1} (d - F1'b1 - F2'b2) -> Sbar \ z = zbar
        //                 |                       |
        //    reuse Cholesky factorization   Inner product calculated above
        getSfactorization;
        chol_SolveInPlace(Sbar, zy, nstates, nstates);
      }
      block.sync();
      // Propagate information to solution vector
      //    y = y - F zbar
    }
  }           
