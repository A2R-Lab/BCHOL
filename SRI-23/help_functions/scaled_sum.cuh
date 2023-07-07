#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__

/*
    Compute the scaled sum of two matrices
    alpha * A + B
    store the result back in B
*/

void scaled_sum(std::uint32_t m,
                std::uint32_t n,
                T alpha,
                T *A,
                T *B,
                cgrps::thread_group g = cgrps::this_thread_block()
               ){
    const unsigned max = m*n;
    uint32_t element, ind, row, col;
    T res;
    for(element = g.thread_rank(); element < max; element += g.size()){
        res = static_cast<T>(0);
        row = element % m;
        col = element / m;
        
        B[col*m + row] += A[col*m + row] * B[col*m + row];
    }
}