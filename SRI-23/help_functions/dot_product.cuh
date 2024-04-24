/*
    Computes dot product of A and B, and sum result to C
    Given A, B, C
    Updates C to alpha A'T B + beta*C 
*/

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void dot_product(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha,
          T *A,
          T *B,
          T beta,
          T *C,
          cgrps::thread_group g)
{
    const unsigned max = m*k;
    uint32_t element, ind, row, col;
    T res;
    for(element = g.thread_rank(); element < max; element += g.size()){
        res = static_cast<T>(0);
        row = element % m;
        col = element / m;

        for(ind = 0; ind < n; ind++){
            res += A[row*n + ind] * B[col*n + ind];
        }
        C[col*m + row] = alpha * res + beta * C[col*m + row];
    }
}

