import sys

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
    C = solver.data.ndlqr_GetNdFactor(k, 0)
    F = solver.fact.ndlqr_GetNdFactor(k,0)
    z = solver.soln.ndlqr_GetNdFactor(k,0)
    Q = solver.diagonals[2*k]
    R = solver.diagonals[2*k+1]


    #solve the block systems of eqiations
    # [   -I   ] [Fy]   [Cy]   [ 0 ]    [-A'    ]
    # [-I  Q   ] [Fx] = [Cx] = [ A'] => [ 0     ]
    # [      R ] [Fu]   [Cu]   [ B']    [ R \ B']
    F.lambdas = np.matrix.copy(C.state) 
    F.lambdas = (F.lambdas*-1)
    r = len(F.state)
    c = len(F.state[0])
    F.state = [[0.0 for j in range(c)] for i in range(r)]
    F.input = np.matrix.copy(C.input)
    Rchol = ndlqr_GetRFactorizon(solver.cholfacts, 0)
    Rchol = returnCholesky(R)
    MatrixCholeskySolveWithInfo(R,F.input,Rchol) #Fu = R\Cu
    MatrixCholeskySolveWithInfo(R, z.input, Rchol)
   
    #Rchol = lianlg.cholesky(R) What is this doing here?
    zy_temp = np.matrix.copy(C.lambdas) # grab an unused portion of the matrix data
    zy_temp=np.matrix.copy(z.lambdas) #why do we immediately change the matrix?
    z.lambdas = np.matrix.copy(z.state) #MatrixCopy(&z->lambda, &z->state);
    z.lambdas = (-1)*(Q@zy_temp)+z.lambdas # MatrixMultiply(Q, &zy_temp, &z->lambda, 0, 0, -1.0,  -1.0);  // zy = - Q * zy - zx
    z.state =np.matrix.copy(zy_temp)
    z.state = z.state*(-1)
    Qchol = solver.cholfacts
    Qchol = returnCholesky(Q) 

  else: #line 60 in srs
    level = 0

    Q = solver.diagonals[2 * k]
    Qchol =ndlqr_GetQFactorizon(solver.cholfacts,k) #check how we implemented GetQFactorization
    Qchol = returnCholesky(Q)

    z = solver.soln.ndlqr_GetNdFactor(k,0)

    #All the terms that don't apply at the last time step
    if (k < nhorizon - 1):
      level = solver.tree.ndlqr_GetIndexLevel(k)
      C = solver.data.ndlqr_GetNdFactor(k, level)
      F = solver.fact.ndlqr_GetNdFactor(k, level)

      R = solver.diagonals[2 * k + 1];
      C = ndlqr_GetRFactorizon(solver.cholfacts, k)
      Rchol = returnCholesky(R) 

      MatrixCholeskySolveWithInfo(R, z.input, Rchol) # solve zu = R \ zu  (R \ -r)
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
    prev_level = solver.tree.ndlqr_GetIndexLevel(k - 1)
    C = solver.data.ndlqr_GetNdFactor(k, prev_level)
    F = solver.fact.ndlqr_GetNdFactor(k, prev_level)
    F.state = np.matrix.copy(C.state)#MatrixCopy(&F->state, &C->state);  // the -I matrix
    MatrixCholeskySolveWithInfo(Q, F.state,Qchol)  # solve Q \ -I from previous time step
    MatrixSetConst(F.input, 0.0) #   // Initialize the B2 matrix to zeros
  return 0

"""
@brief Calculates one of the inner products needed at level @p data_level
 
Calculates the following:

 \f[
 \bar{\Lambda_{k+1}^{(p)}} = \bar{\Lambda_k^{(p)}}S +
    \langle X_k^{(j)}, \bar{X}_k^{(p)} \rangle +
    \langle U_k^{(j)}, \bar{U}_k^{(p)} \rangle +
   \langle X_{k+1}^{(j)}, \bar{X}_{k+1}^{(p)} \rangle +
    \langle U_{k+1}^{(j)}, \bar{U}_{k+1}^{(p)} \rangle
  \f]
 
  where \f$ j \f$ is @p data_level, \f$ p \f$ is @p fact_level, and \f$ k \f$ is
  @p index.
 
  @param data       The data for the original KKT matrix
  @param fact       The current data for the factorization
  @param index      Knot point index
  @param data_level Level index for the original data, or equivalently the current level
                    being processed by the outer solve
  @param fact_level Level index for the factorization data, or equivalently the parent or
                    upper level. @p fact_level >= @p data_level.
  @return 0 if successful
"""
def ndlqr_FactorInnerProduct(data,fact,index,data_level,fact_level):
  C1 = data.ndlqr_GetNdFactor(index, data_level)
  F1 = fact.ndlqr_GetNdFactor(index,fact_level)
  C2 = data.ndlqr_GetNdFactor(index+1,data_level)
  F2 = fact.ndlqr_GetNdFactor(index+1,fact_level)
  S = F2.ndlqr_GetLambdaFactor()
  S = C1.state@F1.state
  S = C1.input@F1.state + S
  S = C2.state@F2.state + S
  S = C2.input@F2.input
  #return S?


def ndlqr_SolveCholeskyFactor(fact,cholinfo,index,level,upper_level):
  if(fact == None):
    return -1
  if (upper_level<= level):
    sys.stdout.write("ERROR: `upper_level` must be greater than `level`.")
  F = fact.ndlqr_GetNdFactor(index+1,level)
  Sbar = F.lambdas
  G = fact.ndlqr_GetNdFactor(index+1,upper_level)
  f = G.lambdas
  MatrixCholeskySolveWithInfo(Sbar,f, cholinfo)
  #return?

def ndlqr_UpdateShurFactor(fact,soln,index,i,level,upper_level,calc_lambda):
  if(fact==None or soln == None): #if (!fact || !soln) is it correct?
    return -1
  f_factor = soln.ndlqr_GetNdFactor(index+1,upper_level)
  g = soln.ndlqr_GetNdFactor(i , upper_level)
  F = fact.ndlqr_GetNdFactor(i , level)
  f = f_factor.lambdas
  if(calc_lambda):
    g.state = (-1)*(F.state@f )+ g.lambdas
  g.state = (-1)*(F.state@f )+ g.lambdas
  g.input = (-1)*(F.state@f )+ g.lambdas
  #return 0 ?


def ndlqr_ShouldCalcLambda(tree,index,i):
  node = tree.node_list + index
  is_start = i == node.left_inds_start or i == node.right_inds_start
  return not is_start or i == 0

def ndlqr_ComputeShurCompliment(solver,index,level,upper_level):
  node = solver.tree.node_list + index
  left_start = node.left_inds_start
  right_stop = node.right_inds_stop
  fact = solver.fact
  if upper_level == 0:
    soln = solver.soln
  else:
    soln = solver.fact
  for i in range(left_start, right_stop + 1):
    i += 1
    calc_lambda = ndlqr_ShouldCalcLambda(solver.tree, index, i)
    ndlqr_UpdateShurFactor(fact, soln, index, i, level, upper_level, calc_lambda)
