#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;
#include <cmath>
#include "lowerBackSub.cuh"

/**
 * @brief Perform a Cholesky decomposition in place.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A. .
 *
 * @param T* s_A: a square symmetric matrix , col major order.
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */

template <typename T> 
__device__
void chol_InPlace(uint32_t n, T *s_A,cgrps::thread_group g = cgrps::this_thread_block())
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (uint32_t row = 0; row < n-1; row++) {
        // square root
        if (ind == 0) {
            s_A[n*row+row] = pow(s_A[n*row+row], 0.5);
        }
        __syncthreads();

        // normalization
        for (uint32_t k = ind+row+1; k < n; k+= stride) {
            s_A[n*row+k] /= s_A[n*row+row];
        }
        __syncthreads();
        
        // inner prod subtraction
        for(uint32_t j = ind+row+1; j < n; j+= stride) {
            for (uint32_t k = 0; k < row+1; k++) {
                s_A[n*(row+1)+j] -= s_A[n*k+j]*s_A[n*k+row+1];
            }
        }
        __syncthreads();
    }
    if (ind == 0) {
        s_A[n*n-1] = pow(s_A[n*n-1], 0.5);
    }
}

/**
 * @brief Perform a Cholesky decomposition in place.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A. .
 *
 * @param T* s_A: a square symmetric matrix, row major
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */


template <typename T>
__device__
void chol_InPlace_r(uint32_t n,
                  T *s_A,
                  cgrps::thread_group g = cgrps::this_thread_block()) {
    
    for (unsigned col = 0; col < n; col++) {
        if (g.thread_rank() == 0){
            T sum = 0;
            T val = s_A[n*col+col];
            for(unsigned col_l = 0 ; col_l < col; col_l++) {
                sum += pow(s_A[col*n+col_l],2);
            }
            s_A[col*n+col] = sqrt(val - sum);

        }
        g.sync();
        for(unsigned row = g.thread_rank() + col + 1; row < n; row += g.size()){
            T sum = 0;
            T val = s_A[row*n+col];
            for(unsigned k = 0; k < col; k++) {
                sum += s_A[row*n+k]*s_A[col*n+k];
            }
            s_A[row*n+col] = (1.0/s_A[col*n+col])*(s_A[row*n+col]-sum);
        }
        g.sync();
    }
}

 /**
 *  @brief Solves linear system of equations with cholesky matrix.
 *
 * Calls a lowerBackSub on Cholesky decomposed matrix @p s_A, and a vector @p s_b , storing the solution  in the
 * @p s_b. .
 *
 * @param T* s_A:  Cholesky decomposition of a matrix A, row major
 * @param  T* s_b:  a vector/matrix that completes the linear system 
 * @param bool istransposed: a boolean to check if the matrix is transposed
 * @param int n : number of cols/rows
 * @param int m; number of cols/rows
 *
 */
template <typename T>
__device__
void cholSolve_InPlace(T *s_A, T *s_b, bool istransposed, int n, int m, cgrps::thread_group g = cgrps::this_thread_block()) {
    lowerBackSub_InPlace<T>(s_A, s_b, 0, n, m, g);
    lowerBackSub_InPlace<T>(s_A, s_b, 1, n, m, g);
}
