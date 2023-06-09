#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#includ <cmath>
#include <cooperative_groups.h>
#include "./GLASS/GLASS.cuh"

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
void solveLeaf(unit32_t index,
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
               //cgrps::thread_group g?) {
               
    int k = index;
    zy_temp[];
    set_const(nstates*nstates, zy_temp, 0);
               
    if (index ==0) {
        Q = &Q_R;
        R = &Q_R[nstates*nstates]; //double check
               
        //Solve the block system of equations.
        glas::copy(nstates*nstates, s_A_B, s_F_lambda); 
        glas::scal(nstates*nstates,-1.0, s_F_lambda); //-A'_0
        set_const(nstates*nstates,s_F_state, 0); 
        glas::copy(nstates*ninputs,s_A_B[nstates*nstates], s_F_input); //copy  B_0
        glass:chol_InPlace(ninputs, R);
        cholSolve_InPlace(R, s_F_input, ninputs, nstates); //ASK XIAN! Fu = R\B
        cholSolve_InPlace(R, s_q_r[nstates], ninputs, 1);  //zu = R\zu

        //Solve the block system of eqn (!overwriting d and q_r vectors!)
        glas::copy(nstates,d,zy_temp); 
        glas::copy(nstates,q_r, d);
        glas::gemv(nstates,nstates,-1.0, Q, zy_temp, -1.0, d);  // zy = - Q * zy - zx
        glass:copy(zy_temp,q_r);
        glass::scal(nstates,-1.0,q_r); // zx = -zy
        glass:chol_InPlace(nstates,Q) //not sure if that's ok to have it inplace
    
      } else {
          
        int level = 0;
        Q = &Q_R[0]; 
        glass:chol_InPlace(nstates,Q);
        
        //Not the last timestep
        if(k<nhorizon -1) {
            //what to do with level? do I just continue to use k?
            R = &Q_R[nstaes*nstates];
            glass:chol_InPlace(ninputs,R);
            cholSolve_InPlace(R, q_r[nstates], ninputs,1); //zu = R \ zu 

            glass::copy(A_B[0],F_state);
            cholSolve_InPlace(Q,F_state,nstates,nstates);

            glass::copy(A_B[nstates*nstates],F_input);
            cholSolve_InPlace(R,F_input,ninput, nstates);  //DOUBLE CHECK!
                   
           //initialize all F matrices of index+1 to 0 - not sure if it should be in solveLeaf
           set_const(nstates*nstates, 0,  F_lambda[(nstates*nstates)]); 
           set_const(nstates*nstates, 0,  F_state[(nstates*nstates)]);
           set_const(nstates*nstates, 0,  F_input[(nstates*nstates)]);
           //Initialize with -Identity matrix
           diag_Matrix_set(nstates*nstates,-1 , F_state[(nstates*nstates)]);
        }

        //Only the last timestep
        cholSolve_InPlace(Q,q_r[0]);
         
        
        //int prev_level = ndlqr_GetIndexLevel(&solver->tree,k-1); ??
         // NOTE: This is -I on the state for explicit integration
          //For implicit integrators we'd use the A2, B2 partials wrt the next
          //state and control
        //update k
        
        cholSolve_InPlace(Q,F_state); //solve Q \ -I from previous time step
        }
    }
}

template <typename T> 
__device__
void factorInnerProduct() {
}

template <typename T> 
__device__
void solveCholFactor() {
}

template <typename T> 
__device__
void updateShur() {
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
           T *d_d)
//what is exit_tol?
{
    //Ask Emre about cgrps again!
    const cgrps::thread_block block = cgrps::this_thread_block();	 
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const unit32_t states_sq = nstates*nstates;
    const unit32_t inputs_sq = ninputs*ninputs;
    const unit32_t inputs_sq = ninputs*ninputs;
    const unit32_t inp_states = nstates*ninputs;
    const uint32_t depth = log2(nhorizon);
    
           
    //move everything to shared memory?
    extern __shared__ T s_temp[];
    T *s_Q_R = s_temp;
    T *s_q_r = s_Q_R + (states_sq + input_sq)*nhorizon;
    T *s_A_B = s_q_r + (ninputs+nstates)*nhorizon;
    T *s_d = s_A_B + (states_sq+inp_states)*nhorizon;
    T *s_F_lambda = s_d + (nstates*nhorizon);
    T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
    T *s_F_input = s_F_state + (states_sq *nhorizon* depth);
           
    //check if you need soln vector or put it in a different Kernel?
    //check if you need to transpose A_B
           
    //negate q_r,d vectors 
    for (std::uint32_t ind = ; ind < (ninput+nstates)*nhorizon; ind+=block_dim){
               s_q_r[ind] *= -1;
    }
    for (std::uint32_t ind = ; ind < (nstates)*nhorizon; ind+=block_dim){
               s_d[ind] *= -1;
    }
           
    //prob initialize -I and 0 matrices?

    //should solveLeaf in parallel
    for (std::uint32_t ind = ; ind < nhorizon; ind+=block_dim) {
           solveleaf(ind, nstates, ninputs, nhorizon, s_Q_R+ind*(states_sq+input_sq),
                     s_q_r+ind*(ninputs+nstates), s_A_B+ind*(states_sq+inp_states),
                     s_d+ind*nstates, s_F_lambda, s_F_state, s_Finput);
                     //s_F_lambda+ind*(states_sq), s_F_state+ind*(states_sq) for F matrices calculate level too!
    }
  
  
  
  
}
           
