#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/* @brief convert vector v_Q to diagonal matrix n*n m_Q
*  @params:
*  unit32_t  n lenght of vector v_Q
*  T *v_Q vector of length n
*  T *m_Q matrix n*n to change the diagonal
*/


template <typename T>
__device__
void diag_Matrix_set(T *v_Q, T *m_Q, unit32_t  n, cgrps::thread_group g) {
    
    for(uint32_t ind = g.thread_rank(); ind < n; ind+= g.size()){
        m_Q[ind*n+ind] = v_Q[n];        
    }
    
}
