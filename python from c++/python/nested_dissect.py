import numpy as np
import math

def initBTlevel(nhorizon):
    depth = int(np.log2(nhorizon))
    levels = -np.ones(nhorizon,dtype=int)
    for level in range (depth):
        start = 2 **level-1
        step = 2 **(level+1)
        for i in range (start,nhorizon,step):
            levels[i]=level
    return levels

def solveLeaf(levels,index, nstates,ninputs,nhorizon,Q,R,q,r,A,B,d,F_lambda,F_state,F_input):
    #solve independent equations
    level = levels[index]
    lin_index = index+nhorizon*level
    if(index ==0):
        F_lambda[0]=A[0]*-1
        F_input[0]= B[0]
        R[0]=np.linalg.cholesky(R[0])
        F_input[0]=np.linalg.solve(R[0].T,
                                   np.linalg.solve(R[0],F_input[0].T)).T
        r[0]=np.linalg.solve()


        




