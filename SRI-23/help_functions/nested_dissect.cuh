#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
// should we put into one header file?

#include "./copy_mult.cuh"
#include "./chol_InPlace.cuh"
#include "./chol_SolveInPlace.cuh"
#include "./diag_Matrix_set.cuh"
#include "./set_const.cuh"
#include "./dot_product.cuh"
#include "./scaled_sum.cuh"
#include "./print_debug.cuh"

/** @brief Solve all the equations for the lower-level diagonal blocks, by timestep.
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
__device__ void solveLeaf(int *s_levels,
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

  // const for KKT matrix
  const uint32_t states_sq = nstates * nstates;
  const uint32_t inputs_sq = ninputs * ninputs;
  const uint32_t inp_states = ninputs * nstates;
  const uint32_t cost_step = states_sq + inputs_sq;
  const uint32_t dyn_step = states_sq + inp_states;

  // setting up arrays for specific indexes
  int level = s_levels[index];
  float *q = &s_q_r[index * (ninputs + nstates)];
  float *r = &s_q_r[index * (ninputs + nstates) + nstates];
  float *A = &s_A_B[index * dyn_step];
  float *B = &s_A_B[index * dyn_step + states_sq];
  float *d = &s_d[index * nstates];

  // factorized matrices
  float *F_lambda = &s_F_lambda[(index + nhorizon * level) * states_sq];
  float *F_state = &s_F_state[(index + nhorizon * level) * states_sq];
  float *F_input = &s_F_input[(index + nhorizon * level) * inp_states];
  float *zy_temp;

  if (index == 0)
  {
    float *Q = &s_Q_R[cost_step * index];
    float *R = &s_Q_R[cost_step * index + states_sq];

    // Solve the block system of equations.
    glass::copy<float>(nstates * nstates, -1.0, A, F_lambda, cgrps::this_thread_block());

    // dont need to coz we initialized with 0s set_const(nstates*nstates,0, s_F_state);
    glass::copy<float>(nstates * ninputs, 1.0, B, F_input); // copy  B_0
    chol_InPlace<float>(ninputs, R);
    __syncthreads();
    cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates); // Fu = R\B
    cholSolve_InPlace<float>(R, r, false, ninputs, 1);             // zu = R\zu

    // Solve the block system of eqn (!overwriting d and q_r vectors!)
    zy_temp = &s_F_state[nhorizon * states_sq];
    glass::copy<float>(nstates, 1.0, d, zy_temp);
    glass::copy<float>(nstates, 1.0, q, d);
    __syncthreads();
    glass::gemv<float>(nstates, nstates, -1.0, Q, zy_temp, -1.0, d); // zy = - Q * zy - zx
    __syncthreads();
    glass::copy<float>(nstates, -1.0, zy_temp, q);
    __syncthreads();
    set_const<float>(nstates, 0.0, zy_temp); // initialize back to 0s
    chol_InPlace<float>(nstates, Q);
  }
  else
  {

    float *Q = &s_Q_R[index * cost_step];
    chol_InPlace<float>(nstates, Q, cgrps::this_thread_block());

    // Not the last timestep
    if (index < nhorizon - 1)
    {
      float *R = &s_Q_R[index * cost_step + states_sq];
      chol_InPlace<float>(ninputs, R, cgrps::this_thread_block());
      __syncthreads();
      cholSolve_InPlace<float>(R, r, false, ninputs, 1); // zu = R \ zu

      glass::copy<float>(nstates * nstates, A, F_state);
      __syncthreads();
      cholSolve_InPlace<float>(Q, F_state, false, nstates, nstates);
      glass::copy<float>(ninputs * nstates, 1.0, B, F_input);
      __syncthreads();
      cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates);
    }

    // Only term at the last timestep
    cholSolve_InPlace<float>(Q, q, false, nstates, 1); // Q\-q
    int prev_level = s_levels[index - 1];
    float *F_state_prev = s_F_state + (index + nhorizon * prev_level) * states_sq; // prev level  F_state
    diag_Matrix_set<float>(nstates, -1.0, F_state_prev);
    __syncthreads();
    cholSolve_InPlace<float>(Q, F_state_prev, false, nstates, nstates); // solve Q \ -I from previous time step
  }
}

/** @brief Calculates one of the inner products needed at current level
 *
 * Calculates the following:
 *
 * \f[
 * \bar{\Lambda_{k+1}^{(p)}} = \bar{\Lambda_k^{(p)}}S +
 *   \langle X_k^{(j)}, \bar{X}_k^{(p)} \rangle +
 *   \langle U_k^{(j)}, \bar{U}_k^{(p)} \rangle +
 *   \langle X_{k+1}^{(j)}, \bar{X}_{k+1}^{(p)} \rangle +
 *   \langle U_{k+1}^{(j)}, \bar{U}_{k+1}^{(p)} \rangle
 * \f]
 *
 * where \f$ j \f$ is @level, \f$ p \f$ is @p fact_level, and \f$ k \f$ is
 * @p index.
 *
 * @param s_A_B                                     The data for the original KKT matrix
 * @param fact_state, fact_input, fact_lambda       The current data for the factorization
 * @param index                                     Knot point index
 * @param fact_level                               Level index for the factorization data, or equivalently the parent or
 *                   upper level. @p fact_level >= current level.
 * @param nstates, ninputs,nhorizon                KKT parameters
 */
template <typename T>
__device__ void factorInnerProduct(T *s_A_B,
                                   T *fact_state,
                                   T *fact_input,
                                   T *fact_lambda,
                                   int index,
                                   int fact_level,
                                   uint32_t nstates,
                                   uint32_t ninputs,
                                   uint32_t nhorizon)
{

  int dyn_step = nstates * nstates + nstates * ninputs;

  float *C1_state = s_A_B + (index * dyn_step);
  float *C1_input = s_A_B + (index * dyn_step + nstates * nstates);

  uint32_t linear_index = index + nhorizon * fact_level;
  float *F1_state = fact_state + linear_index * (nstates * nstates);
  float *F1_input = fact_input + linear_index * (ninputs * nstates);

  linear_index = (index + 1) + nhorizon * fact_level;
  float *F2_state = fact_state + linear_index * (nstates * nstates);
  float *S = fact_lambda + linear_index * (nstates * nstates); // F2_lambda

  dot_product<float>(nstates, nstates, nstates, 1.0, C1_state, F1_state, -1.0, S, cgrps::this_thread_block()); // S = C1x'F1x
  dot_product<float>(nstates, ninputs, nstates, 1.0, C1_input, F1_input, 1.0, S, cgrps::this_thread_block());
  __syncthreads();
  scaled_sum<float>(nstates, nstates, -1.0, F2_state, S, cgrps::this_thread_block()); // equivalent to -I'F2_state
}

/* @brief Calculates one of the inner products needed at current level, for solution vector instead of the factorized data matrices
 *
 * Calculates the following:
 *
 * \f[
 * \bar{\Lambda_{k+1}^{(p)}} = \bar{\Lambda_k^{(p)}}S +
 *   \langle X_k^{(j)}, \bar{X}_k^{(p)} \rangle +
 *   \langle U_k^{(j)}, \bar{U}_k^{(p)} \rangle +
 *   \langle X_{k+1}^{(j)}, \bar{X}_{k+1}^{(p)} \rangle +
 *   \langle U_{k+1}^{(j)}, \bar{U}_{k+1}^{(p)} \rangle
 * \f]
 *
 * where \f$ j \f$ is @level, \f$ p \f$ is @p fact_level, and \f$ k \f$ is
 * @p index.
 *
 * @param s_A_B                                     The data for the original KKT matrix
 * @param s_q_r, s_d       The current data for the factorization
 * @param index                                     Knot point index
 * @param fact_level                               Level index for the factorization data, or equivalently the parent or
 *                   upper level. @p fact_level >= current level.
 * @param nstates, ninputs,nhorizon                KKT parameters
 */
template <typename T>
__device__ void factorInnerProduct_sol(T *s_A_B,
                                       T *s_q_r,
                                       T *s_d,
                                       int index,
                                       uint32_t nstates,
                                       uint32_t ninputs,
                                       uint32_t nhorizon)
{

  int dyn_step_AB = nstates * nstates + nstates * ninputs;
  int dyn_step_qr = nstates + ninputs;

  // Matrix
  float *C1_state = s_A_B + (index * dyn_step_AB);
  float *C1_input = s_A_B + (index * dyn_step_AB + nstates * nstates);

  // Vector
  float *F1_state = s_q_r + (index * dyn_step_qr);
  float *F1_input = s_q_r + (index * dyn_step_qr + nstates);

  index += 1;
  float *F2_state = s_q_r + (index * dyn_step_qr);
  float *S = s_d + (index * nstates);

  dot_product<float>(nstates, nstates, 1, 1.0, C1_state, F1_state, -1.0, S, cgrps::this_thread_block()); // S = C1x'F1x
  dot_product<float>(nstates, ninputs, 1, 1.0, C1_input, F1_input, 1.0, S, cgrps::this_thread_block());
  __syncthreads();
  scaled_sum<float>(nstates, 1, -1.0, F2_state, S, cgrps::this_thread_block()); // equivalent to -I'F2_state
}

/** @brief Use the precomputed Cholesky factorization to solve for y at each parent level
 *Solve the following linear system of equations, overwriting the right-hand-side.
 *
 * \f[
 * \bar{\Lambda}_{k+1}^{(j)} y = \bar{\Lambda}_{k+1}^{(p)}
 * \f]
 *
 * where \f$ j \f$ is @p level, \f$ p \f$ is @p upper_level, and \f$ k \f$ is
 * @p index.
 *
 * This is the same as solving
 * \f[
 * -\bar{B}_i^{(j)} y_i^{(j,p)} = \bar{b}_i^{(j,p)}
 * \f]
 * using the notation from the paper.
 *
 * @param fact_state, fact_input,fact_lambda        Data for the factorization
 * @param index       Knot point index
 * @param level       Level index for the level currently being processed by the
 *                    upper-level solve.
 * @param upper_level Level index for the right-hand-side. @p upper_level > @p level.
 */
template <typename T>
__device__ void SolveCholeskyFactor(T *fact_lambda,
                                    int index,
                                    int level,
                                    int upper_level,
                                    int nstates,
                                    int ninputs,
                                    int nhorizon)
{
  float *Sbar = fact_lambda + ((index + 1) + nhorizon * level) * nstates * nstates;
  float *f = fact_lambda + ((index + 1) + nhorizon * upper_level) * nstates * nstates;

  cholSolve_InPlace<float>(Sbar, f, 0, nstates, nstates);
}

/** @brief gets the index in "level" that corresponds to 'index'
 * @param nhorizon KKT constant
 * @param depth KKT constant
 * @param level current level
 * @param i index
 * @param levels an array that surves as a binary tree structure, returns level for each index
 */
__device__ int getIndexFromLevel(int nhorizon, int depth, int level, int i, int *levels)
{
  int num_nodes = pow(2, depth - level - 1);
  int leaf = i * num_nodes / nhorizon;
  int count = 0;
  for (int k = 0; k < nhorizon; k++)
  {
    if (levels[k] != level)
    {
      continue;
    }
    if (count == leaf)
    {
      return k;
    }
    count++;
  }
  return -1;
}

/** @brief Determines if the \f$ \Lambda \f$ should be updated during
 *        ndlqr_UpdateSchurFactor()
 *
 * Basically checks to see if the \f$ \Lambda \f$ for the given index is a $B_i^{(p)}$ for
 * any level greater than or equal to the current level.
 *
 * @param index     Knot point index calculated using GetIndexFromLevel()
 * @param i         Knot point index being processed
 * @param nhorizon  KKT constant
 * @param levels    an array that surves as a binary tree structure, returns level for each index
 */
__device__ bool shouldCalcLambda(int index, int i, int nhorizon, int *levels)
{
  int left_start = index - int(pow(2, levels[index])) + 1;
  int right_start = index + 1;
  bool is_start = i == left_start || i == right_start;
  return !is_start || i == 0;
}

/** @brief Calculates \f$ x \f$ and \f$ z \f$ to complete the factorization at the current
 *        level
 *
 * Calculates the following
 *
 * \f[
 * \bar{\Lambda}_k^{(p)} =  \bar{\Lambda}_k^{(p)} - \bar{\Lambda}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \bar{X}_k^{(p)} =  \bar{X}_k^{(p)} - \bar{X}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \bar{U}_k^{(p)} =  \bar{U}_k^{(p)} - \bar{U}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \f]
 *
 * where \f$ \bar{\Lambda}_{k_\text{sep}}^{(p)} \f$ is equivalent to $y_i^{(j,p)}$ from the
 * paper and is the result of SolveCholeskyFactor() for @p index, @p level, and
 * @p upper_level.
 *
 * @param fact_state,
 * fact_input, fact_lambda  Pointers for the factorization data
 * @param index             Knot point index of the "separator" at level @p level.
 *                          Should be calculated using getIndexFromlevel().
 * @param i                 Knot point index to be processed
 * @param level             Level index for the level currently being processed
 * @param upper_level       Level index of the upper level.
 *                          @p upper_level > @p level
 * @param calc_lambda       Whether the \f$ \Lambda \f$ factor should be updated.
 *                          This should be calculated using shouldCalcLambda().
 * @param nstates,ninputs,
 * nhorizon                 KKT constants
 */
template <typename T>
__device__ void updateShur(T *fact_state,
                           T *fact_input,
                           T *fact_lambda,
                           int index,
                           int i,
                           int level,
                           int upper_level,
                           bool calc_lambda,
                           int nstates,
                           int ninputs,
                           int nhorizon)
{
  float *g_state;
  float *g_input;
  float *g_lambda;

  float *F_state;
  float *F_input;
  float *F_lambda;

  int linear_index = (index + 1) + nhorizon * upper_level;
  float *f = fact_lambda + (linear_index * nstates * nstates);

  linear_index = i + nhorizon * upper_level;
  g_state = fact_state + (linear_index * (nstates * nstates));
  g_input = fact_input + (linear_index * (nstates * ninputs));
  g_lambda = fact_lambda + (linear_index * nstates * nstates);

  linear_index = i + nhorizon * level;
  F_state = fact_state + linear_index * nstates * nstates;
  F_input = fact_input + linear_index * nstates * ninputs;
  F_lambda = fact_lambda + linear_index * nstates * nstates;

  if (calc_lambda)
  {
    glass::gemm<float, 0>(nstates, nstates, nstates, -1.0, F_lambda, f, 1.0, g_lambda, cgrps::this_thread_block());
  }
  glass::gemm<float, 0>(nstates, nstates, nstates, -1.0, F_state, f, 1.0, g_state, cgrps::this_thread_block());
  glass::gemm<float, 0>(ninputs, nstates, nstates, -1.0, F_input, f, 1.0, g_input, cgrps::this_thread_block());
}
/** @brief Calculates \f$ x \f$ and \f$ z \f$ to complete the factorization at the current
 *        level for solution vector instead of factorization matrices
 *
 * Calculates the following
 *
 * \f[
 * \bar{\Lambda}_k^{(p)} =  \bar{\Lambda}_k^{(p)} - \bar{\Lambda}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \bar{X}_k^{(p)} =  \bar{X}_k^{(p)} - \bar{X}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \bar{U}_k^{(p)} =  \bar{U}_k^{(p)} - \bar{U}_k^{(j)}
 * \bar{\Lambda}_{k_\text{sep}}^{(p)} \f]
 *
 * where \f$ \bar{\Lambda}_{k_\text{sep}}^{(p)} \f$ is equivalent to $y_i^{(j,p)}$ from the
 * paper and is the result of SolveCholeskyFactor() for @p index, @p level, and
 * @p upper_level.
 *
 * @param fact_state,
 * fact_input, fact_lambda  Pointers for the factorization data
 * @param index             Knot point index of the "separator" at level @p level.
 * @param s_q_r, s_d        Solution vector
 *                          Should be calculated using getIndexFromlevel().
 * @param i                 Knot point index to be processed
 * @param level             Level index for the level currently being processed
 * @param upper_level       Level index of the upper level.
 *                          @p upper_level > @p level
 * @param calc_lambda       Whether the \f$ \Lambda \f$ factor should be updated.
 *                          This should be calculated using shouldCalcLambda().
 * @param nstates,ninputs,
 * nhorizon                 KKT constants
 */
template <typename T>
__device__ void updateShur_sol(T *fact_state,
                               T *fact_input,
                               T *fact_lambda,
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
  float *g_state;
  float *g_input;
  float *g_lambda;

  float *F_state;
  float *F_input;
  float *F_lambda;

  // ndlqr_GetNdFactor(soln, index + 1, 0, &f_factor);
  // Matrix* f = &f_factor->lambda;
  float *f = s_d + ((index + 1) * nstates);

  // ndlqr_GetNdFactor(soln, i, 0, &g);
  int dyn_step_qr = nstates + ninputs;
  g_state = s_q_r + (i * dyn_step_qr);
  g_input = s_q_r + (i * dyn_step_qr + nstates);
  g_lambda = s_d + (i * nstates);

  // from fact, looks at level
  int linear_index = i + nhorizon * level;
  F_state = fact_state + linear_index * nstates * nstates;
  F_input = fact_input + linear_index * nstates * ninputs;
  F_lambda = fact_lambda + linear_index * nstates * nstates;

  if (calc_lambda)
  {
    glass::gemm<float, 0>(nstates, nstates, 1, -1.0, F_lambda, f, 1.0, g_lambda, cgrps::this_thread_block());
  }
  glass::gemm<float, 0>(nstates, nstates, 1, -1.0, F_state, f, 1.0, g_state, cgrps::this_thread_block());
  glass::gemm<float, 0>(ninputs, nstates, 1, -1.0, F_input, f, 1.0, g_input, cgrps::this_thread_block());
}
