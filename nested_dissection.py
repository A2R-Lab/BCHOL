def MatrixSetConst(A, const):
  for i in A:
    for j in A[i]:
      A[i][j] = const 

def ndlqr_SolveLeaf(solver, index):
  """
  Solve all the equations for the lowest-level diagonal blocks, by timestep
  """
  Qchol = None #choleskyinfo
  Rchol = None #choleskyinfo

  nstates = solver.nstates
  nhorizon = solver.nhorizon
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
    MatrixSetConst(F.state, 0.0)
    F.input = np.matrix.copy(C.input)
    Rchol = solver.cholinfo.ndlqr_GetRFactorizon(0)
    Rchol =  #IMPLEMENT!
    MatrixCholeskySolveWithInfo(R,F.input,Rchol) #Fu = R\Cu
    MatrixCholeskySolveWithInfo(R, z.input, Rchol)
   
    #Rchol = lianlg.cholesky(R) What is this doing here?
    zy_temp = np.matrix.copy(C.lambda) # grab an unused portion of the matrix data
    zy_temp=np.matrix.copy(z.lambda) #why do we immediately change the matrix?
    z.lambda = np.matrix.copy(z.state) #MatrixCopy(&z->lambda, &z->state);
    z.lambda = -Q*zy_temp # MatrixMultiply(Q, &zy_temp, &z->lambda, 0, 0, -1.0,  -1.0);  // zy = - Q * zy - zx
    z.state =np.matrix.copy(zy_temp)
    z.state = z.state*(-1)
    Qchol = solver.cholfacts
    MatrixCholeskyFactorizeWithInfo(Q, Qchol) #IMPLEMENT

  else: #line 60 in srs
    level = 0

    Q = solver.diagonals[2 * k]
    Qchol =ndlqr_GetQFactorizon(solver.cholfacts,k) #check how we implemented GetQFactorization
    MatrixCholeskyFactorizeWithInfo(Q, Qchol) #Implement

    z = ndlqr_GetNdFactor(solver.soln,k,0)

    #All the terms that don't apply at the last time step
    if (k < nhorizon - 1):
      level = ndlqr_GetIndexLevel(solver.tree, k)
      ndlqr_GetNdFactor(solver->data, k, level, &C);
      ndlqr_GetNdFactor(solver->fact, k, level, &F);

      R = &solver->diagonals[2 * k + 1];
      C = ndlqr_GetRFactorizon(solver.cholfacts, k)
      MatrixCholeskyFactorizeWithInfo(R, Rchol) #IMPLEMENT

      MatrixCholeskySolveWithInfo(R, z.input Rchol) # solve zu = R \ zu  (R \ -r)
      F.state = np.matrix.copy(C.state)
      MatrixCholeskySolveWithInfo(Q, F.state,Qchol) # solve Fx = Q \ Cx  (Q \ A')
      F.input =np.matrix.copy(C.input)
      MatrixCholeskySolveWithInfo(R, F.input,Rchol) # solve Fu = Q \ Cu  (R \ B')
    #// Only term at the last time step
    MatrixCholeskySolveWithInfo(Q, z.state,Qchol) # solve zx = Q \ zx  (Q \ -q)

    """
    Solve for the terms from the dynamics of the previous time step
    // NOTE: This is -I on the state for explicit integration
    //       For implicit integrators we'd use the A2, B2 partials wrt the next
    //       state and control"""
    prev_level = ndlqr_GetIndexLevel(solver.tree, k - 1)
    C = ndlqr_GetNdFactor(solver.data, k, prev_level)
    F = ndlqr_GetNdFactor(solver.fact, k, prev_level)
    F.state = np.matrix.copy(C.state)#MatrixCopy(&F->state, &C->state);  // the -I matrix
    MatrixCholeskySolveWithInfo(Q, F.state,Qchol)  # solve Q \ -I from previous time step
    MatrixSetConst(F.input, 0.0) #   // Initialize the B2 matrix to zeros
  return 0

def ndlqr_FactorInnerProduct(data,fact,index,data_level,fact_level):

def ndlqr_SolveCholeskyFactor(fact,cholinfo,index,level,upper_level):

def ndlqr_UpdateShurFactor(fact,soln,index,i,level,upper_level,calc_lambda):

def ndlqr_ShouldCalcLambda(tree,index,i):

def ndlqr_ComputeShurCompliment(solver,index,level,upper_level):


