# Do we even need CholeskyInfo class?

def MatrixCholeskySolve(A, b):
    """
    @brief Solve a linear system using a precomputed Cholesky factorization

    Overwrite the input vector @p b.
    Prefer to use the more robust MatrixCholeskySolveWithInfo().

    @param[in]    A A square matrix whose Cholesky decomposition has already been computed.
    @param[inout] b The right-hand-side vector. Stores the solution vector.
    @return       0 if successful
    """
    out = clap_CholeskySolve(A, b)
    return out

def MatrixCholeskySolveWithInfo(A, b, cholinfo):
    """
    @brief Solve a linear system using a precomputed Cholesky factorization

    @param[in]      A A square matrix whose Cholesky decomposition has already been computed.
    @param[inout]   b The right-hand-side vector. Stores the solution vector.
    @param cholinfo Information about the precomputed Cholesky factorization in @p A.
    @return
    """
    out = 0
    cholinfo = []
    clap_CholeskySolve(A, b)
    return out