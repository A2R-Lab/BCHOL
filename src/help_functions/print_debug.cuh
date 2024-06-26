#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>
/*
* The following funcitons might come handy for debug and printing purposes
- checkEquality - Checks that two 1D arrays are the same and have no NaN values
- printMatrix - Prints the desired matrix in row-column order
- print_KKT - Prints F_lambda,
*/

/** @brief Checks that two 1D arrays are the same
 * @param T *array_a - pointer to the first array
 * @param T *array_b - pointer to the second array
 * @param uint32_t size - number of elements in each array
 */
__host__ __device__ bool checkEquality(float *array_a, float *array_b, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++)
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

/** @brief Prints the facotrizatoin matrices and the solution vectors.
 * @param T *F_lambda, F_state, F_input - pointer to the stored factorization matrices
 * @param T *d, *q_r - pointer to the state and q_r vector / solution vectors.
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 depth - depth of the factorization tree
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */
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
    for (unsigned i = 0; i < nhorizon - 1; i++)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
    // last time step print only d and q vectors
    unsigned i = nhorizon - 1;
    printf("\nd%d: \n", i);
    printMatrix(d + i * nstates, 1, nstates);

    printf("\nq%d: \n", i);
    printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);
}

/** @brief Prints the solution vectors.
 * @param T *d, *q_r - pointer to the state and q_r vector / solution vectors.
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */

template <typename T>
__host__ __device__ void print_soln(T *d, T *q_r, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{
    for (uint32_t i = 0; i < nhorizon - 1; i++)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
    // last time step print only d and q vectors
    unsigned i = nhorizon - 1;
    printf("\nd%d: \n", i);
    printMatrix(d + i * nstates, 1, nstates);

    printf("\nq%d: \n", i);
    printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);
}

/** @brief Prints thethe solution vector at required step
 * @param uint32_t i - index of the knot point
 * @param T *d, *q_r - pointer to the state and q_r vector / solution vectors.
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */

template <typename T>
__host__ __device__ void print_soln_step(uint32_t i, T *d, T *q_r, uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{
    if (i != nhorizon - 1)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);

        printf("\nr%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates) + nstates), 1, ninputs);
    }
    else if (i == nhorizon - 1)
    {
        printf("\nd%d: \n", i);
        printMatrix(d + i * nstates, 1, nstates);

        printf("\nq%d: \n", i);
        printMatrix(q_r + (i * (ninputs + nstates)), 1, nstates);
    }
    else
    {
        printf("i must be between 0 and nhorizon/knot_points-1\n");
    }
}

/** @brief Prints thethe solution vectors from ram and from shared memory, useful for comparison
 * @param T *s_d, *s_q_r - pointer to the state and q_r vector / solution vectors on shared memory.
 * @param T *d_d, *d_q_r - pointer to the state and q_r vector / solution vectors on ram.
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */
template <typename T>
__host__ __device__ void print_soln_ram_shared(T *s_d, T *s_q_r, T *d_d, T *d_q_r,
                                               uint32_t nhorizon, uint32_t nstates, uint32_t ninputs)
{
    for (uint32_t i = 0; i < nhorizon - 1; i++)
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
    // Last time step
    unsigned i = nhorizon - 1;
    printf("\ns_d%d: \n", i);
    printMatrix(s_d + i * nstates, 1, nstates);
    printf("\nd_d%d: \n", i);
    printMatrix(d_d + i * nstates, 1, nstates);

    printf("\ns_q%d: \n", i);
    printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);
    printf("\nd_q%d: \n", i);
    printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);
}

/** @brief Prints the facotrizatoin matrices and the solution vectors from shared memory and from ram.
 * @param T *s_F_lambda, s_F_state, s_F_input - pointer to the stored factorization matrices on shared
 *  * @param T *d_F_lambda, ds_F_state, d_F_input - pointer to the stored factorization matrices on ram
 * @param T *s_d, *s_q_r - pointer to the state and q_r vector / solution vectors on shared
 * @param T *d_d, *d_q_r - pointer to the state and q_r vector / solution vectors on ram
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 depth - depth of the factorization tree
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */
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
    for (unsigned i = 0; i < nhorizon - 1; i++)
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
    // Last time step
    unsigned i = nhorizon - 1;
    printf("\ns_d%d: \n", i);
    printMatrix(s_d + i * nstates, 1, nstates);
    printf("\nd_d%d: \n", i);
    printMatrix(d_d + i * nstates, 1, nstates);

    printf("\ns_q%d: \n", i);
    printMatrix(s_q_r + (i * (ninputs + nstates)), 1, nstates);
    printf("\nd_q%d: \n", i);
    printMatrix(d_q_r + (i * (ninputs + nstates)), 1, nstates);
}

/** @brief Prints the facotrizatoin matrices at the specific step
 * @param T *F_lambda, F_state, F_input - pointer to the stored factorization matrices o
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 depth - depth of the factorization tree
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */
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

/** @brief Prints the facotrizatoin matrices and the solution vectors at the specific step
 * @param T *F_lambda, F_state, F_input - pointer to the stored factorization matrices
 * @param T *d, *q_r - pointer to the state and q_r vector / solution vectors
 * @param uint32 nhorizon - number of knot points
 *  @param uint32 depth - depth of the factorization tree
 *  @param uint32 nstates - dimension of the state vector
 *  @param uint32 ninputs - dimension of the control vector
 * */
template <typename T>
__host__ __device__ void print_step(uint32_t ind, T *F_lambda, T *F_state, T *F_input, T *d,
                                    T *q_r, uint32_t nhorizon, uint32_t depth, uint32_t nstates, uint32_t ninputs)
{
    uint32_t states_sq = nstates * nstates;
    uint32_t inp_states = nstates * ninputs;
    if (ind != nhorizon - 1)
    {
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
    else if (ind==nhorizon -1) {
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
    }
    else {
        printf("Index must be between 0 and %d", nhorizon-1);
    }
}