def PrintComp(base, new):
    print(f"{base} / {new} ({base / new} speedup)")

### Time only
class NdLqrProfile(object):
    """
    @brief A struct describing how long each part of the solve took, in milliseconds.

    ## Methods
    - ndlqr_NewNdLqrProfile() # Srishti - "unnecesaary -> this is init in Python"
    - ndlqr_ResetProfile()
    - ndlqr_CopyProfile()
    - ndlqr_PrintProfile()
    - ndlqr_CompareProfile()
    """
    def _init_(self):
        """
        @brief Create a profile initialized with zeros
        """
        self.t_total_ms = 0.0
        self.t_leaves_ms = 0.0
        self.t_products_ms = 0.0
        self.t_cholesky_ms = 0.0
        self.t_cholsolve_ms = 0.0
        self.t_shur_ms = 0.0
        self.num_threads = -1

    # Srishti - "Commenting this out because unnecesaary -> this is init in Python"
    '''
    def ndlqr_NewNdLqrProfile(self):    
        prof = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]
        return prof
    '''

    def ndlqr_ResetProfile(self):
        """
        @brief Reset the profile to its initialized state

        @param prof A profile
        """
        self.t_total_ms = 0.0
        self.t_leaves_ms = 0.0
        self.t_products_ms = 0.0
        self.t_cholesky_ms = 0.0
        self.t_cholsolve_ms = 0.0
        self.t_shur_ms = 0.0
        return

    def ndlqr_CopyProfile(self, dest, src): # Srishti - "needs to be changed -> self is dest or src, depending on usage, currently no usage seen in Python code"
        """
        @brief Copy the profile information to a new profile

        @param dest New location for data. Existing data will be overwritten.
        @param src Data to be copied.
        """
        dest.num_threads = src.num_threads
        dest.t_total_ms = src.t_total_ms
        dest.t_leaves_ms = src.t_leaves_ms
        dest.t_products_ms = src.t_products_ms
        dest.t_cholesky_ms = src.t_cholesky_ms
        dest.t_cholsolve_ms = src.t_cholsolve_ms
        dest.t_shur_ms = src.t_shur_ms
        return

    def ndlqr_PrintProfile(self, profile):
        """
        @brief Print a summary fo the profile

        @param profile
        """
        print(f"Solved with {profile.num_threads} threads")
        print(f"Solve Total:    {profile.t_total_ms} ms")
        print(f"Solve Leaves:   {profile.t_leaves_ms} ms")
        print(f"Solve Products: {profile.t_products_ms} ms")
        print(f"Solve Cholesky: {profile.t_cholesky_ms} ms")
        print(f"Solve Solve:    {profile.t_cholsolve_ms} ms")
        print(f"Solve Shur:     {profile.t_shur_ms} ms")
        return

    def ndlqr_CompareProfile(self, base, prof): # Srishti - "needs to be changed -> self is base or prof, depending on usage, currently no usage seen in Python code"
        """
        @brief Compare two profiles, printing the comparison to stdout

        @param base The baseline profile
        @param prof The "new" profile
        """
        print(f"Num Threads:     {base.num_threads} / {prof.num_threads}")
        print(f"Solve Total:     ")
        PrintComp(base.t_total_ms, prof.t_total_ms)
        print(f"Solve Leaves:    ")
        PrintComp(base.t_leaves_ms, prof.t_leaves_ms)
        print(f"Solve Products:  ")
        PrintComp(base.t_products_ms, prof.t_products_ms)
        print(f"Solve Cholesky:  ")
        PrintComp(base.t_cholesky_ms, prof.t_cholesky_ms)
        print(f"Solve CholSolve: ")
        PrintComp(base.t_cholsolve_ms, prof.t_cholsolve_ms)
        print(f"Solve Shur Comp: ")
        PrintComp(base.t_shur_ms, prof.t_shur_ms)

class NdLqrSolver(object):
    """
    @brief Main solver for rsLQR

    Core struct for solving problems with rsLQR. Allocates all the required memory
    up front to avoid any dynamic memory allocations at runtime. Right now, the
    horizon length is required to be a power of 2 (e.g. 32,64,128,256,etc.).

    ## Construction and destruction
    Use ndlqr_NewNdLqrSolver() to initialize a new solver. This should always be
    paired with a single call to ndlqr_FreeNdLqrSolver().

    ## Typical Usage

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    LQRProblem* lqrprob = ndlqr_ReadTestLQRProblem();  // your data here
    int nstates = lqrprob->lqrdata[0]->nstates;
    int ninputs = lqrprob->lqrdata[0]->ninputs;
    int nhorizon = lqrprob->nhorizon;

    NdLqrSolver* solver = ndlqr_NewNdLqrSolver(nstates, ninputs, nhorizon);
    ndlqr_InitializeWithLQRProblem(lqrprob, solver);
    ndlqr_Solve(solver);
    ndlqr_PrintSolveSummary();
    ndlqr_FreeLQRProblem(lqrprob);
    ndlqr_FreeNdLqrSolver(solver);
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Methods (Srishti - "this list is probably outdated")
    - ndlqr_NewNdLqrSolver()
    - ndlqr_FreeNdLqrSolver()
    - ndlqr_InitializeWithLQRProblem()
    - ndlqr_Solve()
    - ndlqr_ResetSolver()
    - ndlqr_GetNumVars()
    - ndlqr_SetNumThreads()
    - ndlqr_PrintSolveProfile()
    - ndlqr_GetProfile()
    """
    def _init_(self, nstates, ninputs, nhorizon): # Called `ndlqr_NewNdLqrSolver` in C program
        """
        @brief Create a new solver, allocating all the required memory.

        Must be followed by a later call to ndlqr_FreeNdLqrSolver().

        @param nstates Number of elements in the state vector
        @param ninputs Number of control inputs
        @param nhorizon Length of the time horizon. Must be a power of 2.
        @return A pointer to the new solver
        self.nstates # < size of state vector
        self.ninputs # < number of control inputs
        self.nhorizon # < length of the time horizon
        self.depth # < depth of the binary tree
        self.nvars # < number of decision variables (size of the linear system)
        self.tree
        self.diagonals # < (nhorizon,2) array of diagonal blocks (Q,R)
        self.data # < original matrix data
        self.fact # < factorization
        self.soln # < solution vector (also the initial RHS)
        self.cholfacts
        self.solve_time_ms # < total solve time in milliseconds.
        self.linalg_time_ms
        self.profile
        self.num_threads # < Number of threads used by the solver.
        """
        tree = ndlqr_BuildTree(nhorizon)
        nvars = (2 * nstates + ninputs) * nhorizon - ninputs

        diag_size = (nstates * nstates + ninputs * ninputs) * nhorizon
        diag_data = 
        diagonals = 
        for k in range(nhorizon):
            blocksize = nstates * nstates + ninputs * ninputs
            diagonals[2 * k].rows = nstates
            diagonals[2 * k].cols = nstates
            diagonals[2 * k].data = diag_data + k * blocksize
            diagonals[2 * k + 1].rows = ninputs
            diagonals[2 * k + 1].cols = ninputs
            diagonals[2 * k + 1].data = diag_data + k * blocksize + nstates * nstates
        
        cholfacts = ndlqr_NewCholeskyFactors(tree.depth, nhorizon)

        self.nstates = nstates
        self.ninputs = ninputs
        self.nhorizon = nhorizon
        self.depth = tree.depth
        self.nvars = nvars
        self.tree = tree
        self.diagonals = diagonals
        self.data = ndlqr_NewNdData(nstates, ninputs, nhorizon, nstates)
        self.fact = ndlqr_NewNdData(nstates, ninputs, nhorizon, nstates)
        self.soln = ndlqr_NewNdData(nstates, ninputs, nhorizon, 1)
        self.cholfacts = cholfacts
        self.solve_time_ms = 0.0
        self.linalg_time_ms = 0.0
        self.profile = NdLqrProfile()
        self.num_threads = omp_get_num_procs() // 2

    def ndlqr_ResetSolver(self):
        """
        @brief Resets the rsLQR solver

        Resets all of the data in the solver to how it was when it was first initialized.

        @param solver
        """
        ndlqr_ResetNdData(self.data) # Srishti - inplement ndlqr_ResetNdData (part of nddata)
        ndlqr_ResetNdData(self.fact)
        ndlqr_ResetNdData(self.soln)
        self.profile.ndlqr_ResetProfile()
        for i in range(2 * self.nhorizon):
            MatrixSetConst(self.diagonals[i], 0.0)
        return

    # Srishti - "unnecessary function, commenting it out"
    '''
    def ndlqr_FreeNdLqrSolver(self, solver):
        """
        @brief Deallocates the memory for the solver.

        @param solver An initialized solver.
        @return 0 if successful.
        @post solver == NULL
        """
    '''

    def ndlqr_InitializeWithLQRProblem(self, lqrprob): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Initialize the solver with data from an LQR Problem.

        @pre Solver has already been initialized via ndlqr_NewNdLqrSolver()
        @param lqrprob An initialized LQR problem with the data to be be solved.
        @param solver An initialized solver.
        @return 0 if successful
        """
        nstates = self.nstates
        ninputs = self.ninputs
        if lqrprob.nhorizon != self.nhorizon:
            return -1

        # Create a minux identity matrix for copying into the original matrix
        minus_identity = NewMatrix(nstates, nstates)
        MatrixSetConst(&minus_identity, 0)
        for i in range(nstates)
            MatrixSetElement(minus_identity, i, i, -1)

        # Loop over the knot points, copying the LQR data into the matrix data
        # and populating the right-hand-side vector
        Cfactor = 
        zfactor = 
        ndlqr_GetNdFactor(self.soln, 0, 0, zfactor)
        zfactor.lambda.data = lqrprob.x0
        k = 
        for k in range(self.nhorizon - 1):
            if nstates != lqrprob.lqrdata[k].nstates:
                return -1
            if ninputs != lqrprob.lqrdata[k].ninputs:
                return -1

            # Copy data into C factors and rhs vector from LQR data
            level = ndlqr_GetIndexLevel(self.tree, k)
            ndlqr_GetNdFactor(self.data, k, level, Cfactor)
            ndlqr_GetNdFactor(self.soln, k, 0, zfactor)
            A = [nstates, nstates, lqrprob.lqrdata[k].A]
            B = [nstates, ninputs, lqrprob.lqrdata[k].B]
            MatrixCopyTranspose(Cfactor.state, A)
            MatrixCopyTranspose(Cfactor.input, B)
            zfactor.state.data = lqrprob.lqrdata[k].q
            zfactor.input.data = lqrprob.lqrdata[k].r

            # Copy Q and R into diagonals
            Q = self.diagonals[2 * k]
            R = self.diagonals[2 * k + 1]
            MatrixSetConst(Q, 0)
            MatrixSetConst(R, 0)
            for i in range(nstates):
                MatrixSetElement(Q, i, i, lqrprob.lqrdata[k].Q[i])
            for i in range(ninputs):
                MatrixSetElement(&R, i, i, lqrprob.lqrdata[k].R[i])

            # Next time step
            ndlqr_GetNdFactor(self.data, k + 1, level, Cfactor)
            ndlqr_GetNdFactor(self.soln, k + 1, 0, zfactor)
            Cfactor.state.data = minus_identity.data
            MatrixSetConst(Cfactor.input, 0.0)
            zfactor.lambda.data = lqrprob.lqrdata[k].d

        # Terminal step
        zfactor.state.data = lqrprob.lqrdata[k].q
        Q = self.diagonals[2 * k]
        MatrixSetConst(Q, 0)
        for i in range(nstates):
            MatrixSetElement(Q, i, i, lqrprob.lqrdata[k].Q[i])

        # Negate the entire rhs vector
        for i in range(self.nvars):
            self.soln.data[i] = self.soln.data[i] * (-1)

        return 0

    def ndlqr_PrintSolveSummary(self): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Prints a summary of the solve

        Prints solve time, the residual norm, and the number of theads.

        @pre ndlqr_Solve() has already been called
        @param solver
        """
        print("rsLQR Solve Summary")
        print("-------------------")
        print("  The rsLQR solver is a parallel solver for LQR problems")
        print("  developed by the RExLab at Carnegie Mellon University.\n")
        print(f"  Solve time:  {self.solve_time_ms} ms")
        if kMatrixLinearAlgebraTimingEnabled:
            print(f"  LinAlg time: {self.linalg_time_ms} ms ({100.0 * self.linalg_time_ms / self.solve_time_ms}%% of total)")
        print(f"  Solved with {self.num_threads} threads.")
        print("  ")
        MatrixPrintLinearAlgebraLibrary()
        return

    def ndlqr_GetNumVars(self): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Gets the total number of decision variables for the problem.

        @param solver
        """
        return self.nvars

    def ndlqr_SetNumThreads(self, num_threads): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Set the number of threads to be used during the solve

        Does not guarantee that the specified number of threads will be used.
        To query the actual number of threads used during the solve, use the
        ndlqr_GetNumThreads() function after the solve.

        @param solver rsLQR solver
        @param num_threads requested number of threads
        @return 0 if successful
        """
        if not self:
            return -1
        self.num_threads = num_threads
        return 0

    def ndlqr_GetNumThreads(self): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Get the number of threads used during the rsLQR solve

        @param solver A solver which has already been initialized and solved
        @return number of OpenMP threads used the by solver
        """
        if not self:
            return -1
        return self.num_threads

    def ndlqr_PrintSolveProfile(self): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Prints a summary of how long individual components took

        @pre ndlqr_Solve() has already been called
        @param solver A solver which has already been initialized and solved
        @return 0 if successful
        """
        if not self:
            return -1
        ndlqr_PrintProfile(self.profile)
        return 0

    def ndlqr_GetProfile(self): # Srishti - Usage should be modified according to the modified Python code here
        """
        @brief Ge the internal profile data from a solve

        @param solver A solver which has already been initialized and solved
        @return A profile object containing timing information about the solve
                See NdLqrProfile for more info.
        """
        return self.profile
