import math
import numpy as np
import sys
import time 

"""have not finished implementing the following
import binary_tree 
import nested_dissection 
import utils.h
import solver
"""

#TRANSLATION many functions to python built in fcts:


def isPowerOfTwo(x):
    if x < 1:
        return False
    return math.log2(x) % 1 == 0
"""
PowerOfTwo = math.pow(2, x)
"""


"""
MatrixCopy  - numpy.matrix.copy

"""
#linalg.
"""
"""

#END



#skipping implementation of OMP_TOC and get_work
#DEFINE NdLqrSolver class!!!

#CLASSES

class NdFactor (object):
    def _init_(self):
        self.lambda = 
        self.state =        
        self.input=

class NdData(object):
    def _init_(self):
        self.nstates =


class UnitRange (object):
    """
    @brief Represents a range of consecutive integers

    Abstract representation of the set of consecutive integers from
    UnitRange::start to UnitRange::stop.
    """
    def _init_(self):
        self.start # inclusive
        self.stop # exclusive

class BinaryNode (object):
    """
    @brief One of the nodes in the binary tree for the rsLQR solver
    """
    def _init_(self):
        self.idx # < knot point index
        self.level # < level in the tree
        self.levelidx # < leaf index at the current level
        self.left_inds # < range of knot point indices of all left children #do we need them? (Srishti - "idk")
        self.right_inds # < range of knot point indices of all right children
        self.parent # < parent node
        self.left_child # < left child
        self.right_child # < right child

    
def BuildSubTree(start,len):
        """
        actualy builds the tree,need to implement the function (Srishti - "implemented this now, but not sure if it's the correct translation from C")
        """
        mid = (len + 1) // 2
        new_len = mid - 1
        if len > 1:
            root = start + new_len
            left_root = BuildSubTree(start, new_len)
            right_root = BuildSubTree(start + mid, new_len)
            root.left_child = left_root
            root.right_child = right_root
            left_root.parent = root
            right_root.parent = root
            root.left_inds.start = left_root.left_inds.start
            root.left_inds.stop = left_root.right_inds.stop
            root.right_inds.start = right_root.left_inds.start
            root.right_inds.stop = right_root.right_inds.stop
            root.level = left_root.level + 1
            return root
        else:
            k = start.idx
            start.left_inds.start = k
            start.left_inds.stop = k
            start.right_inds.start = k + 1
            start.right_inds.stop = k + 1
            start.level = 0
            start.left_child = None
            start.right_child = None
            return start

def GetNodeAtLevel(node,index,level):
        if(node.level == level):
            return node
        elif (node.level > level):
            if(index <= node.idx):
                return GetNodeAtLevel(node.left_child, index,level)
            else:
                return GetNodeAtLevel(node.right_child,index,level)
        else:
            return GetNodeAtLevel(node.parent,index,level)
     
class OrderedBinarytree(object):
    """
    @brief The binary tree for the rsLQR solver
    
    Caches useful information to speed up some of the computations during the
    rsLQR solver, mostly around converting form linear knot point indices to
    the hierarchical indexing of the algorithm.
    
    ## Construction and destruction
    A new tree can be constructed solely given the length of the time horizon, which must
    be a power of two. Use ndlqr_BuildTree(N) to build a new tree, which can be
    de-allocated using ndlqr_FreeTree().
    
    ## Methods
    - ndlqr_BuildTree()
    - ndlqr_FreeTree()
    - ndlqr_GetIndexFromLeaf()
    - ndlqr_GetIndexLevel()
    - ndlqr_GetIndexAtLevel()
    """
    def _init_(self,nhorizon): # (Srishti - "why does it have nhorizon as input")
        self.root # (C type: BinaryNode*) < root of the tree. Corresponds to the "middle" knot point.
        self.node_list # (C type: BinaryNode*) < a list of all the nodes, ordered by their knot point index
        self.num_elements # (C type: int) < length of the OrderedBinaryTree::node_list
        self.depth # (C type: int) < total depth of the tree

    
    def ndlqr_BuildTree(self, nhorizon): # (Srishti - "Is N (of C code) = nhorizon (of Python code)? - I think yes, look at .c file")
        """
        @brief Construct a new binary tree for a horizon of length @p N
        
        Must be paired with a corresponding call to ndlqr_FreeTree().
        
        @param  N horizon length. Must be a power of 2.
        @return A new binary tree
        """
        assert(isPowerOfTwo(nhorizon))

        for i in range (nhorizon):
            self.node_list[i].idx = i # (Srishti - "don't we have to make node_list of length `nhorizon` first?")
        self.num_elements = nhorizon
        self.depth = math.log2(nhorizon)
        
        # Build the tree
        self.root = BuildSubTree(node_list, nhorizon - 1)

    #don't need ndlqr_FreeeTree (Srishti - "why not?")
      
    def ndlqr_GetIndexFromLeaf(self, leaf, level):
        """
        @brief Get the knot point index given the leaf index at a given level

        @param tree  An initialized binary tree for the problem horizon
        @param leaf  Leaf index
        @param level Level of tree from which to get the index
        @return
        """
        linear_index = math.pow(2, level) * (2 * leaf + 1) - 1
        return linear_index
    
    def ndlqr_GetIndexLevel(self, index):
        """
        @brief Get the level for a given knot point index

        @param tree  An initialized binary tree for the problem horizon
        @param index Knot point index
        @return      The level for the given index
        """
        return self.node_list[index].level   

    def ndlqr_GetIndexAtLevel(tree, leaf, level):
        """
        @brief Get the index in 'level' that corresponds to `index`.

        If the level is higher than the level of the given index, it's simply the parent
        at that level. If it's lower, then it's the index that's closest to the given one, with
        ties broken by choosing the left (or smaller) of the two.

        @param tree  Precomputed binary tree
        @param index Start index of the search. The result will be the index closest to this
        index.
        @param level The level in which the returned index should belong to.
        @return int  The index closest to the provided one, in the given level. -1 if
        unsucessful.
        """
        if tree = None: # (Srishti - "is this syntax correct for Python")
            return -1
        if index < 0 or index >= tree.num_elements:
            print(f"ERROR: Invalid index ({index}). Should be between 0 and {tree.num_elements - 1}.")
        
        if level < 0 or level >= tree.depth:
            print(f"ERROR: Invalid level ({level}). Should be between 0 and {tree.depth - 1}.")

        node = tree.node_list + index
        if index == tree.num_elements - 1:
            node = tree.node_list + index - 1

        base_node = GetNodeAtLevel(node, index, level) # (Srishti - "check how to define const variable here (syntax)")
        return base_node.idx

def PrintComp(base, new):
    print(f"{base} / {new} ({base / new} speedup)")

### Time only
class NdLqrProfile(object):
    """
    @brief A struct describing how long each part of the solve took, in milliseconds.

    ## Methods
    - ndlqr_NewNdLqrProfile()
    - ndlqr_ResetProfile()
    - ndlqr_CopyProfile()
    - ndlqr_PrintProfile()
    - ndlqr_CompareProfile()
    """
    def _init_(self):
        self.t_total_ms
        self.t_leaves_ms
        self.t_products_ms
        self.t_cholesky_ms
        self.t_cholsolve_ms
        self.t_shur_ms
        self.num_threads

    def ndlqr_NewNdLqrProfile(self):
        """
        @brief Create a profile initialized with zeros
        """
        prof = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]
        return prof

    def ndlqr_ResetProfile(self, prof):
        """
        @brief Reset the profile to its initialized state

        @param prof A profile
        """
        prof.t_total_ms = 0.0
        prof.t_leaves_ms = 0.0
        prof.t_products_ms = 0.0
        prof.t_cholesky_ms = 0.0
        prof.t_cholsolve_ms = 0.0
        prof.t_shur_ms = 0.0
        return

    def ndlqr_CopyProfile(self, dest, src):
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

    def ndlqr_CompareProfile(self, base, prof):
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

###
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
    def _init_(self):
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

    def ndlqr_NewNdLqrSolver(self, nstates, ninputs, nhorizon):
        """
        @brief Create a new solver, allocating all the required memory.

        Must be followed by a later call to ndlqr_FreeNdLqrSolver().

        @param nstates Number of elements in the state vector
        @param ninputs Number of control inputs
        @param nhorizon Length of the time horizon. Must be a power of 2.
        @return A pointer to the new solver
        """
        tree = ndlqr_BuildTree(nhorizon)
        solver = 
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

        solver.nstates = nstates
        solver.ninputs = ninputs
        solver.nhorizon = nhorizon
        solver.depth = tree.depth
        solver.nvars = nvars
        solver.tree = tree
        solver.diagonals = diagonals
        solver.data = ndlqr_NewNdData(nstates, ninputs, nhorizon, nstates)
        solver.fact = ndlqr_NewNdData(nstates, ninputs, nhorizon, nstates)
        solver.soln = ndlqr_NewNdData(nstates, ninputs, nhorizon, 1)
        solver.cholfacts = cholfacts
        solver.solve_time_ms = 0.0
        solver.linalg_time_ms = 0.0
        solver.profile = ndlqr_NewNdLqrProfile()
        solver.num_threads = omp_get_num_procs() // 2
        return solver

    def ndlqr_ResetSolver(self, solver):
        """
        @brief Resets the rsLQR solver

        Resets all of the data in the solver to how it was when it was first initialized.

        @param solver
        """
        ndlqr_ResetNdData(solver.data) # Srishti - inplement ndlqr_ResetNdData (part of nddata)
        ndlqr_ResetNdData(solver.fact)
        ndlqr_ResetNdData(solver.soln)
        ndlqr_ResetProfile(solver.profile)
        for i in range(2 * solver.nhorizon):
            MatrixSetConst(solver.diagonals[i], 0.0)
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

    def ndlqr_InitializeWithLQRProblem(self, lqrprob, solver):
        """
        @brief Initialize the solver with data from an LQR Problem.

        @pre Solver has already been initialized via ndlqr_NewNdLqrSolver()
        @param lqrprob An initialized LQR problem with the data to be be solved.
        @param solver An initialized solver.
        @return 0 if successful
        """
        nstates = solver.nstates
        ninputs = solver.ninputs
        if lqrprob.nhorizon != solver.nhorizon:
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
        ndlqr_GetNdFactor(solver.soln, 0, 0, zfactor)
        zfactor.lambda.data = lqrprob.x0
        k = 
        for k in range(solver.nhorizon - 1):
            if nstates != lqrprob.lqrdata[k].nstates:
                return -1
            if ninputs != lqrprob.lqrdata[k].ninputs:
                return -1

            # Copy data into C factors and rhs vector from LQR data
            level = ndlqr_GetIndexLevel(solver.tree, k)
            ndlqr_GetNdFactor(solver.data, k, level, Cfactor)
            ndlqr_GetNdFactor(solver.soln, k, 0, zfactor)
            A = [nstates, nstates, lqrprob.lqrdata[k].A]
            B = [nstates, ninputs, lqrprob.lqrdata[k].B]
            MatrixCopyTranspose(Cfactor.state, A)
            MatrixCopyTranspose(Cfactor.input, B)
            zfactor.state.data = lqrprob.lqrdata[k].q
            zfactor.input.data = lqrprob.lqrdata[k].r

            # Copy Q and R into diagonals
            Q = solver.diagonals[2 * k]
            R = solver.diagonals[2 * k + 1]
            MatrixSetConst(Q, 0)
            MatrixSetConst(R, 0)
            for i in range(nstates):
                MatrixSetElement(Q, i, i, lqrprob.lqrdata[k].Q[i])
            for i in range(ninputs):
                MatrixSetElement(&R, i, i, lqrprob.lqrdata[k].R[i])

            # Next time step
            ndlqr_GetNdFactor(solver.data, k + 1, level, Cfactor)
            ndlqr_GetNdFactor(solver.soln, k + 1, 0, zfactor)
            Cfactor.state.data = minus_identity.data
            MatrixSetConst(Cfactor.input, 0.0)
            zfactor.lambda.data = lqrprob.lqrdata[k].d

        # Terminal step
        zfactor.state.data = lqrprob.lqrdata[k].q
        Q = solver.diagonals[2 * k]
        MatrixSetConst(Q, 0)
        for i in range(nstates):
            MatrixSetElement(Q, i, i, lqrprob.lqrdata[k].Q[i])

        # Negate the entire rhs vector
        for i in range(solver.nvars):
            solver.soln.data[i] = solver.soln.data[i] * (-1)

        return 0

    def ndlqr_PrintSolveSummary(self, solver):
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
        print(f"  Solve time:  {solver.solve_time_ms} ms")
        if kMatrixLinearAlgebraTimingEnabled:
            print(f"  LinAlg time: {solver.linalg_time_ms} ms ({100.0 * solver.linalg_time_ms / solver.solve_time_ms}%% of total)")
        print(f"  Solved with {solver.num_threads} threads.")
        print("  ")
        MatrixPrintLinearAlgebraLibrary()
        return

    def ndlqr_GetNumVars(self, solver):
        """
        @brief Gets the total number of decision variables for the problem.

        @param solver
        """
        return solver.nvars

    def ndlqr_SetNumThreads(self, solver, num_threads):
        """
        @brief Set the number of threads to be used during the solve

        Does not guarantee that the specified number of threads will be used.
        To query the actual number of threads used during the solve, use the
        ndlqr_GetNumThreads() function after the solve.

        @param solver rsLQR solver
        @param num_threads requested number of threads
        @return 0 if successful
        """
        if not solver:
            return -1
        solver.num_threads = num_threads
        return 0

    def ndlqr_GetNumThreads(self, solver):
        """
        @brief Get the number of threads used during the rsLQR solve

        @param solver A solver which has already been initialized and solved
        @return number of OpenMP threads used the by solver
        """
        if not solver:
            return -1
        return solver.num_threads

    def ndlqr_PrintSolveProfile(self, solver):
        """
        @brief Prints a summary of how long individual components took

        @pre ndlqr_Solve() has already been called
        @param solver A solver which has already been initialized and solved
        @return 0 if successful
        """
        if not solver:
            return -1
        ndlqr_PrintProfile(solver.profile)
        return 0

    def ndlqr_GetProfile(self, solver):
        """
        @brief Ge the internal profile data from a solve

        @param solver A solver which has already been initialized and solved
        @return A profile object containing timing information about the solve
                See NdLqrProfile for more info.
        """
        return solver.profile

def ndlqr_GetNdFactor(fact,index,level,F):
    """
    Retrieve the individual Ndfactor out of the NdData  """
    return

def GetSFactorization(cholfacts,leaf,level):
    """
    Define the function"""
    return cholinfo



def ndlqr_ShouldCalcLambda(tree, index, k):
    """Define the function"""
    return

#ACTUAL ALGORITHM
#Step 1
def ndlqr_SolveLeaf(solver, index):
    """
    part of nexted_dissection
    Solve all the equations for the lowest-level diagonal blocks, by timestep
    """
    nstates = solver.nstates
    nhotizon = solver.nhorizon
    k = index 
    if (index == 0):
        C = ndlqr_GetNdFactor(solver.data,k, 0)
        F = ndlqr_GetNdFactor(solver.fact,k,0)
        z = ndlqr_GetNdFactor(solver.soln,k,0)
    Q = solver.diahonals[2*k]
    R = solver.diagonals[2*k+1]


    #solve the block systems of eqiations
    # [   -I   ] [Fy]   [Cy]   [ 0 ]    [-A'    ]
    # [-I  Q   ] [Fx] = [Cx] = [ A'] => [ 0     ]
    # [      R ] [Fu]   [Cu]   [ B']    [ R \ B']
    F.lambda = np.matrix.copy(C.state)
    F.lambda = (F.lambda*-1)
    F.state = (F.state* 0)
    F.input = np.matrix.copy(C.input)
    Rchol = 


    return

# ! MAIN !
def ndqlr_Solve(solver):
    """
    Returns 0 if the system was solved succeffuly.
    """

    #time the clock here for benchmarks omitted
    depth = solver.depth
    nhorizon = solver.nhorizon

    #Solving for independent diagonal block on the lowest level
    for k in range (nhorizon): #1 to N or 0 to N-1?
        ndlqr_SolveLeaf(solver,k)

    #Solving factorization
    for level in range (depth):
        numleaves = pow(2,depth - level -1) 

        #Calc Inner Products 
        cur_depth = depth-level 
        num_products = numleaves*cur_depth

        for i in range (num_products): 
            leaf = i/cur_depth
            upper_level = level+(i%cur_depth)
            index = solver.tree.ndlqr_GetIndexFromLeaf(leaf ,level) #or have this function outside of class BinaryTree?
            ndlqr_FactorInnerProduct(solver.data,solver.fact,index,level, upper_level)
             
        for leaf in range(numleaves):
            index = solver.tree.ndlqr_GetIndexFromLeaf(leaf,level)
            #get the Sbar Matrix calculated above
            #NdFactor* F
            F=ndlqr_GetNdFactor(solver.fact, index+1,level)
            sbar = F.lambda
            cholinfo = ndlqr_GetSFactorization (sovler.choldfacts,leaf,level)
            MatrixCholeskyFactorizeWithInfo(&Sbar,cholinfo)
            """
            Probably can substitute this whole piece with just Cholesky linalg?
            """

        #Solve with Cholesky factor for f
        upper_levels = cur_depth-1
        num_solves = numleaves*upper_levels
        for i in range (num_solves):
            leaf = i/upper_levels
            upper_level = level+1+(i%upper_levels)
            index = ndlqr_GetIndexFromLeaf(solver.tree,leaf,level)
            #ndlqr_GetSFactorization(solver->cholfacts, leaf, level, &cholinfo);
            #ndlqr_SolveCholeskyFactor(solver->fact, cholinfo, index, level, upper_level);
            """Rewrite it"""

        #Shur compliments - check for numpy library!
        num_factors = nhorizon * upper_levels
        for i in range(num_factors):
            k = i/upper_levels
            upper_level = level+1+(i%upper_levels)
            index = ndlqr_GetIndexAtLevel(solver.tree,k,level)
            calc_lambda = ndlqr_ShouldCalcLambda(solver.tree, index, k)
            ndlqr_UdpateShurFactor(solver.fact, solver.fact, index, k, level, upper_level,calc_lambda) 

        for level in range (depth):
            numleaves = PowerOfTwo(depth-level -1)

        #Calculate inner products with right-hand-side, with the factors computed above

        for leaf in range (numleaves):
            index = ndlqr_GetIndexFromLeaf(solver.tree,leaf,level)

            #Calculate z = d-F'b1-F2'b2
            ndlqr_FactorInnerProduct(solver.data, solver.soln, index,level,0)
        
        #Solve for separator variables with cached Cholesky decomp
        for leaf in range (numleaves):
            index = ndlqr_GetIndexFromLeaf(solver.tree,leaf,level)

        #Get the Sbar Matrix calculated above
        """WHAT IS Sbar Matrix??"""
        f = ndlqr_GetNdFactor(solver.fact,index+1,level)
        z = ndlqr_GetNdFactor(solver.soln,index+1,0)
        Sbar = F.lambda
        zy = z.lambda

        #Solve (S - C1'F1 - C2'F2)^{-1} (d - F1'b1 - F2'b2) -> Sbar \ z = zbar
        cholinfo = ndlqr_GetSFactorization(solver.cholfacts,leaf,level)
        MatrixCholeskyFactorizeWithInfo(cholinfo)

        #propogate information to solution vector
        #y = y- Fzbar
        for k in range (nhorizon):
            index = ndlqr_GetIndexAtLevel(solver.tree,k,level)
            calc_lambda=ndlqr_ShouldCalcLambda(solver.tree,index,k)
            ndlqr_UdpateShurFactor
    return 0 

def ndlqr_GetSolution(solver):
    """
    Returns the solution vector as a simple wrapper around a raw pointer
    which points to the data actually stored by the solver
    """
    soln = solver.nvars,1,solver.soln.data
    return soln 

def ndlqr_CopySolution(solver,soln):
    """
    Copies the solution vector to a user supplied array"""
    