#include <cstdint>
#include <cmath>
#include "lowerBackSub.cuh"

//Solves linear system of equations with Cholesky
template <typename T>
__device__
void cholSolve_InPlace(T *s_A, T *s_b, bool istransposed, int n, int m) {
    lowerBackSub_InPlace<T>(s_A, s_b, 0, n, m);
    lowerBackSub_InPlace<T>(s_A, s_b, 1, n, m);
}
