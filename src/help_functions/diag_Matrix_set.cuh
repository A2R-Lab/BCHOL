#pragma once
#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/** @brief convert vector v_Q to diagonal matrix n*n m_Q
*  @params:
*  std::unit32_t  n lenght of vector v_Q
*  T *v_Q vector of length n
*  T *m_Q matrix n*n to change the diagonal
*/


template <typename T>
__device__
void diag_Matrix_set_v(std::uint32_t  n, T *v_Q, T *m_Q,  cgrps::thread_group g =  cgrps::this_thread_block()) {
    
    for(uint32_t ind = g.thread_rank(); ind < n; ind+= g.size()){
        m_Q[ind*n+ind] = v_Q[n];        
    }
    
}

template <typename T>
__device__
void diag_Matrix_set(std::uint32_t  n, T c, T *m_Q,  cgrps::thread_group g =  cgrps::this_thread_block()) {
    
    for(uint32_t ind = g.thread_rank(); ind < n; ind+= g.size()){
        m_Q[ind*n+ind] = c;        
    }
    
}
