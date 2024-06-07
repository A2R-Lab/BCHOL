import numpy as np
import math
import copy
import scipy.linalg as linalg

#WRITE TEST CASES
def initBTlevel(nhorizon):
    depth = int(np.log2(nhorizon))
    levels = -np.ones(nhorizon,dtype=int)
    for level in range (depth):
        start = 2 **level-1
        step = 2 **(level+1)
        for i in range (start,nhorizon,step):
            levels[i]=level
    return levels


#WRITE TESTS!
def gemv(alpha,A,x,beta,y):
    y +=beta *y
    y+=alpha*np.dot(A,x)

#Write tests
def dot_product(alpha, A, B, beta,C):
    return alpha * np.dot(A.T, B) + beta*C
    

def solveLeaf(levels,index, nstates,ninputs,nhorizon,s_Q,s_R,s_q,s_r,s_A,s_B,s_d,
              s_F_lambda,s_F_state,s_F_input):
    level = levels[index]
    lin_index = index+nhorizon*level
    #setting up array for specific indices
    Q  = s_Q[index]
    R = s_R[index]
    q=s_q[index]
    r = s_r[index]
    A=s_A[index]
    B=s_B[index]
    d=s_d[index]
    F_lambda = s_F_lambda[index+nhorizon*level]
    F_state= s_F_state[index+nhorizon*level]
    F_input = s_F_input[index+nhorizon*level]

    if(index ==0):
        F_lambda =copy.deepcopy(A[0])*-1
        F_input= copy.deepcopy(B[0])
        linalg.cho_factor(R)
        linalg.cho_solve(R,F_input,overwrite_b=True)
        linalg.cho_solve(R,r,overwrite_b=True)
        #solve the block system of eqn overwriting d, q,r
        zy_temp = np.zeros(nstates)
        zy_temp = copy.deepcopy(d)
        d = copy.deepcopy(q)
        gemv(-1,Q,zy_temp,-1,d)
        q=copy.deepcopy(zy_temp)*-1
        zy_temp[:] = 0
        linalg.cho_factor(Q)
    else:
        linalg.cho_factor(Q)
        #not the last timestep
        if(index<nhorizon-1):
            linalg.cho_factor(R)
            linalg.cho_solve(R,r, overwrite_b = True)
            F_state = copy.deepcopy(A)
            linalg.cho_solve(Q,F_state)
            F_input = copy.deepcopy(B)
            linalg.cho_solve(R,F_input,overwrite_b = True)
        linalg.cho_solve(Q,q)
        prev_level = levels[index-1]
        F_state_prev = s_F_state[prev_level*nhorizon+index]
        np.fill_diagonal(F_state_prev,-1)
        linalg.cho_solve(Q,F_state_prev,overwrite_b=True)

#write tests
def factorInnerProduct(s_A,s_B, s_F_state,s_F_input,s_F_lambda,index,
                       fact_level,nhorizon,sol=False):
    C1_state=s_A[index]
    C1_input = s_B[index]
    if sol: 
        F1_state = s_F_state[index+nhorizon*fact_level]
        F1_input = s_F_input[index+nhorizon*fact_level]
        F2_state = s_F_state[(index+1)+nhorizon*fact_level]
        S = s_F_lambda[(index+1)+nhorizon*fact_level]
    else:
        F1_state = s_F_state[index]
        F1_input = s_F_input[index]
        F2_state = s_F_state[(index+1)]
        S = s_F_lambda[(index+1)]
    dot_product(1,C1_state,F1_state,-1,S)
    dot_product(1,C1_input,F1_input,1,S)
    S +=-1*F2_state

#Write tests
def getIndexFromLevel(nhorizon,depth,level,i,levels):
    num_nodes=np.power(2,depth-level-1)
    leaf=i*num_nodes/nhorizon
    count = 0
    for k in range (nhorizon):
        if(levels[k]!=level):
            continue
        if(count==leaf):
            return k
        count+=1
    return -1

def shouldCalcLambda(index, i, nhorizon,levels):
    left_start = index - int(np.power(2,levels[index]))+1
    right_start = index+1
    is_start = i==left_start or i ==right_start
    return not is_start or i==0
#Write tests
def updateShur (s_F_state,s_F_input,s_F_lambda,index,i,level,
                upper_level,calc_lambda,nhorizon,sol = False):
    
    if sol:
        f = s_F_lambda[i+1]
        g_state = s_F_state[i]
        g_input = s_F_input[i]
        g_lambda = s_F_lambda[i]

    else:
        lin_index = index+1+(nhorizon*upper_level)
        f = s_F_lambda[lin_index]
        g_state = s_F_state[i+nhorizon*upper_level]
        g_input = s_F_input[i+nhorizon*upper_level]
        g_lambda = s_F_lambda[i+nhorizon*upper_level]

    F_state = s_F_state[i+nhorizon*level]
    F_input = s_F_input[i+nhorizon*level]
    F_lambda = s_F_lambda[i+nhorizon*level]

    if calc_lambda:
        gemv(-1,F_lambda, f,1,g_lambda)
    gemv(-1,F_state,f,1,g_state)
    gemv(-1,F_input,f,1,g_input)


    





        




