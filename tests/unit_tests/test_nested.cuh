#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <iostream>
#include "../../src/helpf.cuh"
#include "../../src/gpu_assert.cuh"

// Define Kernel wrapers

// Wrapper for solve_leaf
template <typename T>
__global__ void solveLeafKernel(int *s_levels,
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
                                T *s_F_input)
{
    solveLeaf(s_levels, index, nstates, ninputs, nhorizon, s_Q_R,
              s_q_r, s_A_B, s_d, s_F_lambda, s_F_state, s_F_input);
}

// Wrapper for factorInnerProduct
template <typename T>
__global__ void factorInnerProductKernel(T *s_A_B,
                                         T *s_F_state,
                                         T *s_F_input,
                                         T *s_F_lambda,
                                         int index,
                                         int fact_level,
                                         uint32_t nstates,
                                         uint32_t ninputs,
                                         uint32_t nhorizon)
{
    factorInnerProduct(s_A_B, s_F_state, s_F_input, s_F_lambda, index, fact_level,
                       nstates, ninputs, nhorizon);
}

// Wrapper for factorInnerProduct
template <typename T>
__global__ void factorInnerProduct_solKernel(T *s_A_B,
                                             T *s_q_r,
                                             T *s_d,
                                             int index,
                                             uint32_t nstates,
                                             uint32_t ninputs,
                                             uint32_t nhorizon)
{
    factorInnerProduct_sol(s_A_B, s_q_r, s_d, index, nstates, ninputs, nhorizon);
}

// Wrapper for SolveCholeskyFactor
template <typename T>
__global__ void SolveCholeskyFactorKernel(T *s_F_lambda,
                                          int index,
                                          int level,
                                          int upper_level,
                                          int nstates,
                                          int ninputs,
                                          int nhorizon)
{
    SolveCholeskyFactor(s_F_lambda, index, level, upper_level, nstates, ninputs, nhorizon)
}

// Wrapper for updateShur
template <typename T>
__global__ void updateShurKernel(T *s_F_state,
                                 T *s_F_input,
                                 T *s_F_lambda,
                                 int index,
                                 int i,
                                 int level,
                                 int upper_level,
                                 bool calc_lambda,
                                 int nstates,
                                 int ninputs,
                                 int nhorizon)
{
    updateShur(s_F_state, s_F_input, s_F_lambda, index, i, level, upper_level,
               calc_lambda, nstates, ninputs, nhorizon);
}

// Wrapper for updateShur
template <typename T>
__global__ void updateShur_solKernel(T *s_F_state,
                                     T *s_F_input,
                                     T *s_F_lambda,
                                     T *s_q_r,
                                     T *s_d,
                                     int index,
                                     int i,
                                     int level,
                                     bool calc_lambda,
                                     int nstates,
                                     int ninputs,
                                     int nhorizon)
{
    updateShur_sol(s_F_state, s_F_input, s_F_lambda, s_q_r, s_d, index, i, level,
                   calc_lambda, nstates, ninputs, nhorizon);
}

//TESTS for nested_dissect

//Test for solve_leaf
TEST(NestedDissectTest,SolveleafTest){
    //Write a test for solveleaf
}

//Test for factor_inner
TEST(NestedDissectTest,FactorInnerTest){
    //Write a test for factor_inner
}

//Test for factor_inner_sol
TEST(NestedDissectTest,FactorInnerTestSol){
    //Write a test for factor_inner_sol
}

//Test for SolveCholeskyFact
TEST(NestedDissectTest,SolveCholTest){
    //Write a test for solvecholfact
}

//Test for factor_inner
TEST(NestedDissectTest,FactorInnerTest){
    //Write a test for factor_inner
}

//Test for factor_inner_sol
TEST(NestedDissectTest,FactorInnerTestSol){
    //Write a test for factor_inner_sol
}