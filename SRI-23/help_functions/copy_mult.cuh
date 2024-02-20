#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

//can use to copy soln vector s_q_r and s_d
template <typename T>
__device__
void copy2(uint32_t n1,
          T alpha1,
          T *src1, 
          T *dst1,
          uint32_t n2,
          T alpha2,
          T *src2, 
          T *dst2
          )
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind < n1+n2; ind += stride){
        T *dst; T *src; T alpha; int index;
        if (ind < n1){dst = dst1; src = src1; alpha = alpha1; index = ind;}
        else{dst = dst2; src = src2; alpha = alpha2; index = ind -n1;}
        dst[index] = alpha * src[index];
    }
}

//can use to copy s_F_lambda, s_F_input, s_F_state
template <typename T>
__device__
void copy3(uint32_t n1,
          T alpha1,
          T *src1, 
          T *dst1,
          uint32_t n2,
          T alpha2,
          T *src2, 
          T *dst2,
          uint32_t n3,
          T alpha3,
          T *src3, 
          T *dst3
          )
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

     for(; ind < n1+n2+n3; ind += stride){
        T *dst; T *src; T alpha; int index;
        if (ind < n1){dst = dst1; src = src1; alpha = alpha1; index = ind;}
        else if (ind <n3) {dst = dst2; src = src2; alpha = alpha2; index = ind - n1;}
        else {dst = dst3; src = src3; alpha = alpha3; index = ind - n1 - n2;}
        dst[index] = alpha * src[index];
    }
}
