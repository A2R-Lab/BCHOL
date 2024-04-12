#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "../help_functions/print_debug.cuh"
#include "../help_functions/chol_InPlace.cuh"
#include "../../../GLASS/glass.cuh"

template <typename T>
__global__ void test_chol(uint32_t n, T *d_A)
{
    // block/threads initialization
    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t grid_dim = gridDim.x;

    // initialize shared memory
    extern __shared__ T s_temp[];
    T *s_A = s_temp;
    // copy from RAM to shared
    glass::copy(n * n, d_A, s_A, cgrps::this_thread_block());
    block.sync();
        // launch chol
    chol_InPlace(n, s_A);
    block.sync();
    if(block_id ==0&&thread_id==0){
        printMatrix(d_A,n,n);
        block.sync();
        printf("hey\n");
        printMatrix(s_A,n,n);
        block.sync();
    }
    // copy from shared to RAM
    // copy from RAM to shared
    glass::copy(n * n, s_A, d_A, cgrps::this_thread_block());
}