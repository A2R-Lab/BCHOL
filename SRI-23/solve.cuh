#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../GLASS/glass.cuh"
// should we put into one header file?

#include "./help_functions/chol_InPlace.cuh"
#include "./help_functions/chol_SolveInPlace.cuh"
#include "./help_functions/diag_Matrix_set.cuh"
#include "./help_functions/set_const.cuh"
#include "./help_functions/dot_product.cuh"
#include "./help_functions/scaled_sum.cuh"
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
    set_const<float>(nstates * nstates, 0.0, s_F_state);
    glass::copy<float>(nstates * ninputs, 1.0, B, F_input); // copy  B_0
    chol_InPlace<float>(ninputs, R);
    cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates); // Fu = R\B
    cholSolve_InPlace<float>(R, r, false, ninputs, 1);             // zu = R\zu

    // Solve the block system of eqn (!overwriting d and q_r vectors!)
    zy_temp = &s_F_state[nhorizon * states_sq];
    glass::copy<float>(nstates, 1.0, d, zy_temp);
    glass::copy<float>(nstates, 1.0, q, d);
    glass::gemv<float>(nstates, nstates, -1.0, Q, zy_temp, -1.0, d); // zy = - Q * zy - zx
    glass::copy<float>(nstates, -1.0, zy_temp, q);
    set_const<float>(nstates, 0.0, zy_temp); // initialize back to 0s
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
      cholSolve_InPlace<float>(R, r, false, ninputs, 1); // zu = R \ zu

      glass::copy<float>(nstates * nstates, A, F_state);
      cholSolve_InPlace<float>(Q, F_state, false, nstates, nstates);

      glass::copy<float>(ninputs * nstates, 1.0, B, F_input);
      cholSolve_InPlace<float>(R, F_input, false, ninputs, nstates);
    }

    // Only term at the last timestep
    cholSolve_InPlace<float>(Q, q, false, nstates, 1); // Q\-q
    int prev_level = s_levels[index - 1];
    float *F_state_prev = s_F_state + (index + nhorizon * prev_level) * states_sq; // prev level  F_state
    diag_Matrix_set<float>(nstates, -1.0, F_state_prev);
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
__device__ void SolveCholeskyFactor(T *fact_state,
                                    T *fact_input,
                                    T *fact_lambda,
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

/**
 * @brief Determines if the \f$ \Lambda \f$ should be updated during
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
  glass::copy<float>(cost_step * nhorizon, d_Q_R, s_Q_R);
  glass::copy<float>((nstates + ninputs) * nhorizon, -1.0, d_q_r, s_q_r);
  glass::copy<float>((states_sq + inp_states) * nhorizon, d_A_B, s_A_B);
  glass::copy<float>(nstates * nhorizon, -1.0, d_d, s_d);
  glass::copy<float>(states_sq * nhorizon * depth, d_F_lambda, s_F_lambda);
  glass::copy<float>(states_sq * nhorizon * depth, d_F_state, s_F_state);
  glass::copy<float>(inp_states * nhorizon * depth, d_F_input, s_F_input);
  block.sync(); // block or grid?

  diag_Matrix_set<float>(nstates, -1.0, s_nI); // unnessesary need to clean up later
  block.sync();
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
    glass::copy(nhorizon, s_d + (ind * nstates), d_d + (ind * nstates));
    glass::copy(ninputs + nstates, s_q_r + (ind * (ninputs + nstates)), d_q_r + (ind * (ninputs + nstates)));

    // copy Q_R
    glass::copy(states_sq + inputs_sq, s_Q_R + (ind * (states_sq + inputs_sq)),
                d_Q_R + (ind * (states_sq + inputs_sq)));

    if (ind == 0)
    {
      // copy F_lambda&F_input only for ind==0
      glass::copy(states_sq, s_F_lambda, d_F_lambda);
      glass::copy(inp_states, s_F_input, d_F_input);
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
        glass::copy(states_sq, s_F_state + (states_sq * (cur_level * nhorizon + ind)),
                    d_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass::copy(inp_states, s_F_input + (inp_states * (cur_level * nhorizon + ind)),
                    d_F_input + (inp_states * (cur_level * nhorizon + ind)));
      }
    }
  }
  grid.sync();

  // update the shared ONLY of your neighbours in the future, now everything
  for (uint32_t ind = block_id; ind < nhorizon; ind += grid_dim)
  {
    glass::copy<float>((nstates + ninputs) * nhorizon, d_q_r, s_q_r);
    glass::copy<float>(nstates * nhorizon, d_d, s_d);
    /* copying in the beginning of the level loop
    glass::copy<float>(states_sq*nhorizon*depth, d_F_lambda,s_F_lambda);
    glass::copy<float>(states_sq*nhorizon*depth,d_F_state,s_F_state);
    glass::copy<float>(inp_states*nhorizon*depth,d_F_input,s_F_input);
    */
  }
  grid.sync();

  // set to true to check solveLeaf & copying
  if (!DEBUG)
  {
    if (block_id == 0 && thread_id == 0)
    {
      printf("CHECKING D DATA AFTER SOLVE LEAF");
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

  // using only numleaves blocks (two timesteps per block)
  for (uint32_t level = 0; level < depth; level++)
  {

    //before processing make sure you copied all needed info from RAM
    //need to copy ind and ind+1 for all level - depth
    uint32_t count = getValuesAtLevel(nhorizon, s_levels, level, s_tree_result);
    for (uint32_t b_id = block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t cur_level = level; cur_level < depth; cur_level++)
      {
        // copy F_lambda of ind and of ind+1
        glass::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + ind)));
        glass::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + ind)));

        // copying ind+1
        uint32_t next_ind = ind + 1;
        glass::copy(states_sq, d_F_lambda + (states_sq * (cur_level * nhorizon + next_ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + next_ind)));
        glass::copy(states_sq, d_F_state + (states_sq * (cur_level * nhorizon + next_ind)),
                    s_F_state + (states_sq * (cur_level * nhorizon + next_ind)));
        glass::copy(inp_states, d_F_input + (inp_states * (cur_level * nhorizon + next_ind)),
                    s_F_input + (inp_states * (cur_level * nhorizon + next_ind)));
      }
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
    for (uint32_t b_id= block_id; b_id < count; b_id += grid_dim)
    {
      uint32_t ind = s_tree_result[b_id];
      for (uint32_t up_level = level + 1; up_level < depth; up_level++)
      {
        // copy F_lambda of ind and of ind+1
        glass::copy(states_sq, s_F_lambda + (states_sq * (up_level * nhorizon + ind)),
                    d_F_state + (states_sq * (up_level * nhorizon + ind)));
        glass::copy(states_sq, s_F_state + (states_sq * (up_level * nhorizon + ind)),
                    d_F_state + (states_sq * (up_level * nhorizon + ind)));
        glass::copy(inp_states, s_F_input + (inp_states * (up_level * nhorizon + ind)),
                    d_F_input + (inp_states * (up_level * nhorizon + ind)));
        // copying ind+1
        uint32_t next_ind = ind + 1;
        glass::copy(states_sq, s_F_lambda + (states_sq * (up_level * nhorizon + next_ind)),
                    d_F_state + (states_sq * (up_level * nhorizon + next_ind)));
        glass::copy(states_sq, s_F_state + (states_sq * (up_level * nhorizon + next_ind)),
                    d_F_state + (states_sq * (up_level * nhorizon + next_ind)));
        glass::copy(inp_states, s_F_input + (inp_states * (up_level * nhorizon + next_ind)),
                    d_F_input + (inp_states * (up_level * nhorizon + next_ind)));
      }
    }
  }

  // solve for solution vector using the cached factorization

  //HAVEN"T DONE COPYiNG TO FROM RAM FOR SOLN!!
  for (uint32_t level = 0; level < depth; ++level)
  {
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
