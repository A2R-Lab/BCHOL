#include <stdio.h>
#include <iostream>
#include <cmath>

/**
 * @brief Perform a Cholesky decomposition in place.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A.
 *
 * @param T* s_A: a square symmetric matrix 
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */
 
 template <typename T> 
__device__ 
void cholDecomp_InPlace(T *s_A, int n) {
    for (unsigned col = 0; col < n; col++) {
        if (threadIdx.x == 0){
            T sum = 0;
            T val = s_A[n*col+col]; //entry Ljj
            for(unsigned col_l = 0 ; col_l < col; col_l++) {
                sum += pow(s_A[col*n+col_l],2);
            }
            s_A[col*n+col] = sqrt(val - sum);

        }
        __syncthreads(); //here we computed the diagonal entry of the column
        
        // compute the rest of the column
        for(unsigned row = threadIdx.x + col +1; row < n; row += blockDim.x){
            T sum = 0;
            T val = s_A[row*n+col];
            for(unsigned k = 0; k < col; k++) {
                sum += s_A[row*n+k]*s_A[col*n+k];
            }
            s_A[row*n+col] = (1.0/s_A[col*n+col])*(s_A[row*n+col]-sum);
        }
        __syncthreads();
    }
}
