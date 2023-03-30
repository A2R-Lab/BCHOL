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
    Rchol = ndlqr_GetRFactorizon(solver.cholinfo, 0)
   
    Rchol = lianlg.cholesky(R)
    
    MatrixCholeskySolveWithInfo(R, &F->input, Rchol);  // Fu = R \ Cu
    MatrixCholeskySolveWithInfo(R, &z->input, Rchol);  // zu = R \ zu

    Matrix zy_temp = {nstates, 1,
                      C->lambda.data};  // grab an unused portion of the matrix data
    MatrixCopy(&zy_temp, &z->lambda);
    z.state = z.lambda #MatrixCopy(&z->lambda, &z->state);
   zy.temp =Q*z.lambda -z*x# MatrixMultiply(Q, &zy_temp, &z->lambda, 0, 0, -1.0,
                 #  -1.0);  // zy = - Q * zy - zx

   zy_te MatrixCopy(&z->state, &zy_temp);
    MatrixScaleByConst(&z->state, -1.0);  // zx = -zy
    ndlqr_GetQFactorizon(solver->cholfacts, 0, &Qchol);
    MatrixCholeskyFactorizeWithInfo(Q, Qchol);

  } else {
    int level = 0;

    Q = &solver->diagonals[2 * k];
    ndlqr_GetQFactorizon(solver->cholfacts, k, &Qchol);
    MatrixCholeskyFactorizeWithInfo(Q, Qchol);

    ndlqr_GetNdFactor(solver->soln, k, 0, &z);

    // All the terms that don't apply at the last time step
    if (k < nhorizon - 1) {
      level = ndlqr_GetIndexLevel(&solver->tree, k);
      ndlqr_GetNdFactor(solver->data, k, level, &C);
      ndlqr_GetNdFactor(solver->fact, k, level, &F);

      R = &solver->diagonals[2 * k + 1];
      ndlqr_GetRFactorizon(solver->cholfacts, k, &Rchol);
      MatrixCholeskyFactorizeWithInfo(R, Rchol);

      MatrixCholeskySolveWithInfo(R, &z->input,
                                  Rchol);  // solve zu = R \ zu  (R \ -r)
      MatrixCopy(&F->state, &C->state);
      MatrixCholeskySolveWithInfo(Q, &F->state,
                                  Qchol);  // solve Fx = Q \ Cx  (Q \ A')
      MatrixCopy(&F->input, &C->input);
      MatrixCholeskySolveWithInfo(R, &F->input,
                                  Rchol);  // solve Fu = Q \ Cu  (R \ B')
    }
    // Only term at the last time step
    MatrixCholeskySolveWithInfo(Q, &z->state,
                                Qchol);  // solve zx = Q \ zx  (Q \ -q)

    // Solve for the terms from the dynamics of the previous time step
    // NOTE: This is -I on the state for explicit integration
    //       For implicit integrators we'd use the A2, B2 partials wrt the next
    //       state and control
    int prev_level = ndlqr_GetIndexLevel(&solver->tree, k - 1);
    ndlqr_GetNdFactor(solver->data, k, prev_level, &C);
    ndlqr_GetNdFactor(solver->fact, k, prev_level, &F);
    MatrixCopy(&F->state, &C->state);  // the -I matrix
    MatrixCholeskySolveWithInfo(Q, &F->state,
                                Qchol);  // solve Q \ -I from previous time step
    MatrixSetConst(&F->input, 0.0);      // Initialize the B2 matrix to zeros
  }
  return 0;
}


    return

def ndlqr_SolveLeaves(solver):

def ndlqr_FactorInnerProduct(data,fact,index,data_level,fact_level):

def ndlqr_SolveCholeskyFactor(fact,cholinfo,index,level,upper_level):

def ndlqr_UpdateShurFactor(fact,soln,index,i,level,upper_level,calc_lambda):

def ndlqr_ShouldCalcLambda(tree,index,i):

def ndlqr_ComputeShurCompliment(solver,index,level,upper_level):


