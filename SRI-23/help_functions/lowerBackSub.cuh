#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;
#include <cmath>

/* @brief Solves linear system of equations for a lower-trianguler matrix.
*/

//column major version + matrix - matrix args
template <typename T> 
__device__ 
void lowerBackSub_InPlace(T *s_A, T *s_B, bool istransposed, int n, int m, cgrps::thread_group g = cgrps::this_thread_block()) {
    for(uint32_t k = g.thread_rank(); k < m; k+= g.size()){
        if (istransposed) {
            for (int col = n-1; col >= 0; col--) {
                for(int col_l = n-1; col_l > col; col_l--) {
                    s_B[k*n + col] -= s_A[col*n+col_l] * s_B[k*n + col_l];
                }
                s_B[k*n + col] /= s_A[col*n+col];
            }
        } else {
            for (unsigned col = 0; col < n; col++) {
                for(unsigned col_l = 0 ; col_l < col; col_l++) {
                    s_B[k*n + col] -= s_A[col_l*n+col] * s_B[k*n + col_l];
                }
                s_B[k*n + col] /= s_A[col*n+col];
            }
        }
    }
}