#include <cstdint>
#include <cooperative_groups.h>

template <typename T>
__device__
void set_const(std::uint32_t n, 
          T alpha, 
          T *x)
{
    for(std::uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        x[ind] = alpha;
    }
}
