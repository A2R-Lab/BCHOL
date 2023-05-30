#include <stdio.h>
#include <iostream>
#include <cmath>

/* @brief Solves linear system of equations for a lower-trianguler matrix.
*/


template <typename T> 
__device__ 
void lowerBackSub_InPlace(T *s_A, T *s_b, bool istransposed, int n) {
    for (unsigned col = 0; col < n; col++) {
        if (threadIdx.x == 0){
            for(unsigned col_l = 0 ; col_l < col; col_l++) {
                if (istransposed) {
                    s_b[col] -= s_A[col_l*n+col]; 
                }
                else {
                    s_b[col] -=s_A[col*n+col_l];
                }
            }
            s_b[col] /= s_A[col*n+col];            
        }
        __syncthreads(); //here we computed the b_col

        // multiply matrix by b_col
        for(unsigned row = threadIdx.x + col +1; row < n; row += blockDim.x){
           if (istransposed) {
              s_A[col*n+row] = s_A[col*n+row]*s_b[col];
           } else {
              s_A[row*n+col] = s_A[row*n+col]*s_b[col];
           }
        }
        __syncthreads();
    }
}
