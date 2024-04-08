#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
#include "../../../GLASS/glass.cuh"
#include "./../help_functions/print_debug.cuh"
#include "./../help_functions/nested_dissect.cuh"

namespace cgrps = cooperative_groups;

// @brief The rsLQR tester, the main function of the solver

template <typename T>
__global__ void check_equality_kernel(uint32_t nhorizon,
                                      uint32_t ninputs,
                                      uint32_t nstates,
                                      T *d_Q_R,
                                      T *d_q_r,
                                      T *d_A_B,
                                      T *d_d,
                                      T *d_F_lambda,
                                      T *d_F_state,
                                      T *d_F_input)

{
    // block/threads initialization
    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t grid_dim = gridDim.x;
    const uint32_t size_local = 10;
    const uint32_t size = 10;

    // KKT constants
    const uint32_t states_sq = nstates * nstates;
    const uint32_t inputs_sq = ninputs * ninputs;
    const uint32_t inp_states = ninputs * nstates;
    const uint32_t cost_step = states_sq + inputs_sq;
    const uint32_t dyn_step = states_sq + inp_states;
    const uint32_t depth = log2f(nhorizon);

    // initialize shared memory
    extern __shared__ T s_temp[];
    T *s_Q_R = s_temp;
    T *s_q_r = s_Q_R + (cost_step)*nhorizon;
    T *s_A_B = s_q_r + (ninputs + nstates) * nhorizon;
    T *s_d = s_A_B + (dyn_step)*nhorizon;
    T *s_F_lambda = s_d + nstates * nhorizon;
    T *s_F_state = s_F_lambda + (states_sq * nhorizon * depth);
    T *s_F_input = s_F_state + (states_sq * nhorizon * depth);
    T *s_nI = s_F_input + (depth * inp_states * nhorizon);
    int *s_levels = (int *)(s_nI + states_sq);
    int *s_tree_result = (int *)(s_levels + nhorizon);
    // move ram to shared
    copy2<float>(cost_step * nhorizon, 1, d_Q_R, s_Q_R, dyn_step * nhorizon, 1, d_A_B, s_A_B);
    copy2<float>((nstates + ninputs) * nhorizon, -1.0, d_q_r, s_q_r, nstates * nhorizon, -1.0, d_d, s_d);
    copy3<float>(states_sq * nhorizon * depth, 1, d_F_lambda, s_F_lambda, states_sq * nhorizon * depth, 1,
                 d_F_state, s_F_state, inp_states * nhorizon * depth, 1, d_F_input, s_F_input);

    // extern __shared__ T s_temp[];
    // T *s_A = s_temp;
    // T *s_B = s_temp + size;

    // check on easy local examples
    float s_A[size] = {1, 2, 3, 4, 5, 6, 8, 65, 3.43};
    float s_B[size] = {1, 2, 3, 4, 5, 6, 8, 65, 3.43};

    // if (thread_id == THREAD && block_id == BLOCK)
    // {
    //     // should show true
    //     if (checkEquality(s_A, s_B, size))
    //         printf("Equal case passed\n");
    //     else
    //         printf("ERROR Equal case failed\n");

    //     s_B[0] = 2;
    //     // should show false
    //     if (!checkEquality(s_A, s_B, size))
    //         printf("Diff case passed\n");
    //     else
    //         printf("ERROR Diff case failed\n");

    //     // check cases for copied from host
    //     if (checkEquality(s_Q_R, s_Q_R + cost_step, cost_step))
    //         printf("Equal case passed for shared\n");
    //     else
    //         printf("ERROR Equal case failed for shared\n");

    //     // should show false
    //     if (!checkEquality(s_Q_R, s_Q_R + cost_step * (nhorizon - 1), cost_step))
    //         printf("Diff case passed for shared\n");
    //     else
    //         printf(" ERROR Diff case failed for shared\n");
    // }

    if (thread_id == THREAD && block_id == BLOCK)
    {
        // should show true
        if (checkEqual_prl(s_A, s_B, size))
            printf("Equal case passed\n");
        else
            printf("ERROR Equal case failed\n");

        s_B[0] = 2;
        // should show false
        if (!checkEqual_prl(s_A, s_B, size))
            printf("Diff case passed\n");
        else
            printf("ERROR Diff case failed\n");

        // check cases for copied from host
        if (checkEqual_prl(s_Q_R, s_Q_R + cost_step, cost_step))
            printf("Equal case passed for shared\n");
        else
            printf("ERROR Equal case failed for shared\n");

        // should show false
        if (!checkEqual_prl(s_Q_R, s_Q_R + cost_step * (nhorizon - 1), cost_step))
            printf("Diff case passed for shared\n");
        else
            printf(" ERROR Diff case failed for shared\n");
    }
}