#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "../help_functions/chol_InPlace.cuh"

template <typename T>
__global__ void test_chol(uint32_t n, T* s_A) {
    //copy from RAM to shared

    //launch chol
    chol_InPlace(n,s_A);
    
    block.sync();
    //copy from shared to RAM

}