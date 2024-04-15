#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>

/** @brief Prints the desired matrix in row-column order.
 * @param T *matrix - pointer to the stored matrix
 * @param uint32 rows - number of rows in matrix
 * @param uint32 columns - number of columns
 * */
template <typename T>
__host__ __device__ void printMatrix(T *matrix, uint32_t rows, uint32_t cols)
{
    for (unsigned i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f  ", matrix[j * rows + i]);
        }
        printf("\n");
    }
}

/** @brief Checks that two 1D arrays are the same
 * @param T *array_a - pointer to the first array
 * @param T *array_b - pointer to the second array
 * @param uint32_t size - number of elements in each array
 */
__host__ __device__ bool checkEquality(float *array_a, float *array_b, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        float x = array_a[i];
        float y = array_b[i];

        // Check for NaN values
        if (isnan(x) || isnan(y))
            return false;

        // Check for equality within a tolerance
        float d = abs(x - y);
        if (d > 0.001)
            return false;
    }

    return true;
}

template <typename T>
__host__ __device__ bool checkEqual_prl(T *array_a, T *array_b, uint32_t size)
{
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < size)
    {
        T x = array_a[ind];
        T y = array_b[ind];

        // Check for NaN values
        if (isnan(x) || isnan(y))
            return false;

        // Check for equality within a tolerance
        T d = abs(x - y);
        if (d > 0.001)
            return false;
    }

    return true;
}

/** @brief Checks that two matrices are the same
 * @param T *matrix - pointer to the stored matrix
 * @param uint32 rows - number of rows in matrix
 * @param uint32 columns - number of columns
 * */
template <typename T>
__host__ __device__ bool checkEquallity_mine2(T *matrix_a, T *matrix_b, uint32_t n)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for (; ind < n; ind += stride)
    {
        T x = *(matrix_a + ind);
        T y = *(matrix_b + ind);
        if (isnan(x) || isnan(y))
        {
            return false;
        }
        T d = x - y;
        d = sqrt(d * d);
        if (d > 0.001)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
__host__ __device__ void print_KKT(T *F_lambda, T *F_state, T *F_input, T *d,
                                   T *q_r, uint32_t nhorizon, uint32_t depth, uint32_t nstates, uint32_t ninputs)
{
    uint32_t states_sq = nstates * nstates;
    uint32_t inp_states = nstates * ninputs;
    for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
    {
        if (ind % nhorizon == 0)
        {
            printf("\nLEVEL %d\n", ind / nhorizon);
        }
        printf("\nF_lambda #%d: \n", ind);
        printMatrix(F_lambda + (ind * states_sq), nstates, nstates);

        printf("\nF_state #%d: \n", ind);
        printMatrix(F_state + (ind * states_sq), nstates, nstates);

        printf("\nF_input #%d: \n", ind);
        printMatrix(F_input + ind * inp_states, ninputs, nstates);
    }
    for (unsigned i = 0; i < nhorizon; i++)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
}

template <typename T>
__host__ __device__ void print_soln(T *d, T *q_r, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{
    for (unsigned i = 0; i < nhorizon; i++)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
}

template <typename T>
__host__ __device__ void print_soln_step(uint32_t i, T *d, T *q_r, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{

    printf("\nd%d: \n", i);
    printMatrix(d + i * nstates, 1, nstates);

    printf("\nq%d: \n", i);
    printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

    printf("\nr%d: \n", i);
    printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
}

template <typename T>
__host__ __device__ void print_soln_ram_shared(T *s_d, T *s_q_r, T *d_d, T *d_q_r,
                                               uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{
    for (unsigned i = 0; i < nhorizon; i++)
    {
        printf("\ns_d%d: \n", i);
        printMatrix(s_d + i * nstates, 1, nstates);
        printf("\nd_d%d: \n", i);
        printMatrix(d_d + i * nstates, 1, nstates);

        printf("\ns_q%d: \n", i);
        printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);
        printf("\nd_q%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\ns_r%d: \n", i);
        printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        printf("\nd_r%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
}
template <typename T>
__host__ __device__ void print_ram_shared(T *s_F_lambda, T *s_F_state, T *s_F_input, T *s_d, T *s_q_r,
                                          T *d_F_lambda, T *d_F_state, T *d_F_input, T *d_d, T *d_q_r,
                                          uint32_t nhorizon, uint32_t depth, uint32_t nstates, uint32_t ninputs)
{
    uint32_t states_sq = nstates * nstates;
    uint32_t inp_states = nstates * ninputs;
    for (uint32_t ind = 0; ind < nhorizon * depth; ind++)
    {
        if (ind % nhorizon == 0)
        {
            printf("\nLEVEL %d\n", ind / nhorizon);
        }
        printf("\ns_F_lambda #%d: \n", ind);
        printMatrix(s_F_lambda + (ind * states_sq), nstates, nstates);
        printf("\nd_F_lambda #%d: \n", ind);
        printMatrix(d_F_lambda + (ind * states_sq), nstates, nstates);

        printf("\ns_F_state #%d: \n", ind);
        printMatrix(s_F_state + (ind * states_sq), nstates, nstates);
        printf("\nd_F_state #%d: \n", ind);
        printMatrix(d_F_state + (ind * states_sq), nstates, nstates);

        printf("\ns_F_input #%d: \n", ind);
        printMatrix(s_F_input + ind * inp_states, ninputs, nstates);
        printf("\nd_F_input #%d: \n", ind);
        printMatrix(d_F_input + ind * inp_states, ninputs, nstates);
    }
    for (unsigned i = 0; i < nhorizon; i++)
    {
        printf("\ns_d%d: \n", i);
        printMatrix(s_d + i * nstates, 1, nstates);
        printf("\nd_d%d: \n", i);
        printMatrix(d_d + i * nstates, 1, nstates);

        printf("\ns_q%d: \n", i);
        printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);
        printf("\nd_q%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\ns_r%d: \n", i);
        printMatrix(s_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
        printf("\nd_r%d: \n", i);
        printMatrix(d_q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
}
template <typename T>
__host__ __device__ void print_step_matrix(uint32_t ind, T *F_lambda, T *F_state, T *F_input, uint32_t nstates, uint32_t ninputs)
{
    uint32_t states_sq = nstates * nstates;
    uint32_t inp_states = nstates * ninputs;
    printf("\nF_lambda #%d: \n", ind);
    printMatrix(F_lambda + (ind * states_sq), nstates, nstates);

    printf("\nF_state #%d: \n", ind);
    printMatrix(F_state + (ind * states_sq), nstates, nstates);

    printf("\nF_input #%d: \n", ind);
    printMatrix(F_input + ind * inp_states, ninputs, nstates);
}

template <typename T>
__host__ __device__ void print_step(uint32_t ind, T *F_lambda, T *F_state, T *F_input, T *d,
                                    T *q_r, uint32_t nhorizon, uint32_t depth, uint32_t nstates, uint32_t ninputs)
{
    uint32_t states_sq = nstates * nstates;
    uint32_t inp_states = nstates * ninputs;
    printf("\nF_lambda #%d: \n", ind);
    printMatrix(F_lambda + (ind * states_sq), nstates, nstates);

    printf("\nF_state #%d: \n", ind);
    printMatrix(F_state + (ind * states_sq), nstates, nstates);

    printf("\nF_input #%d: \n", ind);
    printMatrix(F_input + ind * inp_states, ninputs, nstates);
    printf("\nd%d: \n", ind);
    printMatrix(d + ind * nstates, 1, nstates);

    printf("\nq%d: \n", ind);
    printMatrix(q_r + (ind * (ninputs + nstates)), 1, nstates);

    printf("\nr%d: \n", ind);
    printMatrix(q_r + (ind * (ninputs + nstates) + nstates), 1, ninputs);
}