#pragma once
#include <cstdint>
#include <cooperative_groups.h>

template <typename T>
__device__
void add_epsln(std::uint32_t n, T *A)
{
    for(std::uint32_t ind = threadIdx.x; ind < n; ind += blockDim.x){
        A[ind*n+ind]+= 0.000001;
    }
}
