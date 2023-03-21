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

#TRANSLATION many functions to python built in fcts:"""

def isPowerOfTwo(x):
    if x < 1:
        return False
    return math.log2(x) % 1 == 0
"""
PowerOfTwo = math.pow(2, x)"""


"""
MatrixCopy  - numpy.matrix.copy

"""
#linalg.
"""
"""

#END



#skipping implementation of UnitRange OMP_TOC and get_work
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

class BinaryNode (object):
    """
    One of the nodes in the binary tree for the rsLQR solver
    """
    def _init_(self):
        self.idx
        self.level
        self.levelidx
        self.left_inds  #do we need them?
        self.right_inds 
        self.parent
        self.left_child
        self.right_child

    
def BuildSubTree(start,len):
        """
        actualy builds the tree,need to implement the function
        """
        return

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
    def _init_(self,nhorizon):

        self.root
        self.node_list
        self.num_elements
        self.depth

    
    def ndlqr_BuildTree(self, nhorizon):
        assert(isPowerOfTwo(nhorizon))

        for i in range (nhorizon):
            self.node_list[i].idx = i
        self.num_elements = nhorizon
        self.node_list =
        self.depth = math.log2(nhorizon)
        self.root = BuildSubTree(node_list,nhorizon-1)

    #don't need ndlqr_FreeeTree
      
    def ndlqr_GetIndexFromLeaf(self, leaf,level):
        """
        Get the knot point index given the leaf index at a given level
        """
        linear_index = math.pow(2,level)*(2*leaf+1)-1
        return linear_index
    
    def ndlqr_GetIndexLevel(self,index):
        """
        Get the level for a given know point index
        """
        return self.node_list[index].level   

    def ndlqr_GetIndexAtLevel(tree,leaf,level):
        """
        Get the index in 'level' that corresponds to 'index'
        """
        return 


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

#REWRITE USING NUMPY LINALG!:
def ndlqr_FactorInnerProduct(data,fact,index,level,upper_level):
    """
    Define the funtion
    """
    return
def ndlqr_GetSFactorization():
    """LinAlg rewrite!"""

def ndlqr_SolveCholeskyFactor():
    """LinAlg Rewrite!"""

def MatrixCholeskyFactorizeWithInfo():
    """LinAlfRewrite!"""

def ndlqr_UdpateShurFactor():
    """LinAlg rewrite!"""
    return 

#ACTUAL ALGORITHM

#Step 1
def ndlqr_SolveLeaf(solver, index):
    """
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
    