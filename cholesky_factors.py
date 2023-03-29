class NdLqrCholeskyFactors(object):
#DOUBLE CHECK IT! what cholinfo actually contains?


#rewrote ndlqr_NewCholeskyFactors
def _init_(self,depth, nhorizon):
    if(depth<=0):
        return None  #can we return None at _init_ function or should we omit this code?
    if(nhorizon <= 0):
        return None 
    num_leaf_factors = 2 * nhorizon
    num_S_factors = 0
    for level in range (depth) :
        numleaves = math.pow(depth-level-1)
        num_S_factors += numleaves
    numfacts = num_leaf_factors + num_S_factors
    self.depth = depth
    self.nhorizon = nhorizon
    self.cholinfo = None 
    self.numfacts = None



"""
ndlqr_GetQFactorization
"""
def ndlqr_GetQFactorization:

"""
ndlqr_GetRFactorization
"""

"""
ndlqr_GetSFactorization
"""