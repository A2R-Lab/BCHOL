#Still need to check - clap_CholeskySolve

"""
clap_MatrixAddition 
clap_MatrixScale
clap_MatrixMultiply
clap_SymmetricMatrixMultiply
clap_AddDiagonal 

DONT NEED TO WRITE FUNCTIONS IN PYTHON FOR THESE OPERATIONS
"""

"""
clap_CholeskyFactorize - just call lianlg.cholesky(a)
 - will raise an erroe if it can't decompose
 """

def clap_CholeskySolve(L, b):
    # Implements: clap_LowerTriBackSub(L, b, 0)
    scipy.linalg.solve_triangular(L, b, trans=0, overwrite_b=True)
    # Implements: clap_LowerTriBackSub(L, b, 1)
    scipy.linalg.solve_triangular(L, b, trans=1, overwrite_b=True)
    return 0
