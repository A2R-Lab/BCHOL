#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#includ <cmath>
#include <cooperative_groups.h>

namespace cgrps = cooperative_groups;

template <typename T> 
__device__
void solveLeaf() {
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
void solve(std::uint32_t nhorizon,    
           std::uint32_t ninputs,
           std::uint32_t nstates,
           T *d_x0,
           T *d_Q_R,
           T *d_q_r,
           T *d_A_B,
           T *d_d)
//what is exit_tol?
{
  //move everything to shared memory?
  std::uint32_t depth = log2(nhorizon);
  
  //should solveLeaf in parallel
  for (std::uint32_t ind = ...; ind < depth; ++ind) {
    solveleaf(ind)  
  }
  
  
  
  
}
           
