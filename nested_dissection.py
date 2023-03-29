def ndlqr_SolveLeaf(solver, index):
    """
    Solve all the equations for the lowest-level diagonal blocks, by timestep
    """
    Qchol = None 
    Rchol = None 

    nstates = solver.nstates
    nhotizon = solver.nhorizon
    k = index 
    if (index == 0):
        C = ndlqr_GetNdFactor(solver.data,k, 0)
        F = ndlqr_GetNdFactor(solver.fact,k,0)
        z = ndlqr_GetNdFactor(solver.soln,k,0)
    Q = solver.diagonals[2*k]
    R = solver.diagonals[2*k+1]


    #solve the block systems of eqiations
    # [   -I   ] [Fy]   [Cy]   [ 0 ]    [-A'    ]
    # [-I  Q   ] [Fx] = [Cx] = [ A'] => [ 0     ]
    # [      R ] [Fu]   [Cu]   [ B']    [ R \ B']
    F.lambda = np.matrix.copy(C.state) 
    F.lambda = (F.lambda*-1)
    F.state = (np.zeros(())) #Do we know the dimensions of F matrix
    F.input = np.matrix.copy(C.input)
    Rchol = 


    return

def ndlqr_SolveLeaves(solver):

def ndlqr_FactorInnerProduct(data,fact,index,data_level,fact_level):

def ndlqr_SolveCholeskyFactor(fact,cholinfo,index,level,upper_level):

def ndlqr_UpdateShurFactor(fact,soln,index,i,level,upper_level,calc_lambda):

def ndlqr_ShouldCalcLambda(tree,index,i):

def ndlqr_ComputeShurCompliment(solver,index,level,upper_level):


