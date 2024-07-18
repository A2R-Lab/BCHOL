import numpy as np
import math
import copy
import scipy.linalg as linalg
from scipy.linalg.blas import dgemv,dgemm

def is_choleskysafe(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
    
#checked
def initBTlevel(nhorizon):
    depth = int(np.log2(nhorizon))
    levels = -np.ones(nhorizon,dtype=int)
    for level in range (depth):
        start = 2 **level-1
        step = 2 **(level+1)
        for i in range (start,nhorizon,step):
            levels[i]=level
    return levels

def  getValuesAtLevel(binarytree,level):
    index_dict = {}
    for index, value in enumerate(binarytree):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index)
    return index_dict.get(level, []) 
    
    
#Ready to test - 
def solveLeaf(levels,index, nstates,nhorizon,s_Q,s_R,s_q,s_r,s_A,s_B,s_d,
              s_F_lambda,s_F_state,s_F_input):
    level = levels[index]
    lin_index = index+nhorizon*level
    #setting up array for specific indices
    A=s_A[index]
    B=s_B[index]
    Q  = s_Q[index]
    r = s_r[index]
    q = s_q[index]
    d = s_d[index]


    if(index ==0):
        R = s_R[index]
        s_F_lambda[lin_index] =np.copy(A)*-1
        s_F_input[lin_index]=np.copy(B)*1
        F_input=s_F_input[lin_index]
        R,lower_R = linalg.cho_factor(R,lower = True)
        F_input[:]=linalg.cho_solve((R,lower_R),F_input,overwrite_b=True)
        r[:]=linalg.cho_solve((R,lower_R),r,overwrite_b=True)
        #solve the block system of eqn overwriting d, q,r
        zy_temp = np.zeros(nstates)
        zy_temp=np.copy(d)*1
        d = np.copy(q)*1
        s_d[index]=dgemv(-1,Q,zy_temp,beta=-1,y=d,overwrite_y = True)
        s_q[index]=np.copy(zy_temp)*-1
        zy_temp[:] = 0
        Q,lower_Q=linalg.cho_factor(Q,lower=True)
    
    else:
        Q,lower_Q=linalg.cho_factor(Q,lower=True)
        #not the last timestep
        if(index<nhorizon-1):
            R = s_R[index]
            R,lower_R = linalg.cho_factor(R,lower =True)
            r[:]=linalg.cho_solve((R,lower_R),r, overwrite_b = True)
            s_F_state[lin_index] = np.copy(A)*1 
            F_state= s_F_state[lin_index]
            F_state[:]=linalg.cho_solve((Q,lower_Q),F_state,overwrite_b=True)
            s_F_input[lin_index] = np.copy(B)*1 
            F_input = s_F_input[lin_index]
            #CHECK IF YOU NEED TO TRANSPOSE!
            F_input[:]=linalg.cho_solve((R,lower_R),F_input,overwrite_b = True)
        
        q[:]=linalg.cho_solve((Q,lower_Q),q,overwrite_b=True)
        prev_level = levels[index-1]
        F_state_prev = s_F_state[prev_level*nhorizon+index]
        np.fill_diagonal(F_state_prev,-1)
        F_state_prev[:]=linalg.cho_solve((Q,lower_Q),F_state_prev,overwrite_b=True)
        

#write tests
def factorInnerProduct(s_A,s_B, s_F_state,s_F_input,s_F_lambda,index,
                       fact_level,nhorizon,sol=False):
    C1_state=s_A[index]
    C1_input = s_B[index]
    if sol: 
        F1_state = s_F_state[index]
        F1_input = s_F_input[index]
        F2_state = s_F_state[(index+1)]
        S = s_F_lambda[(index+1)]
    else:
        lin_ind = index+(nhorizon*fact_level)
        F1_state = s_F_state[lin_ind]
        F1_input = s_F_input[lin_ind]
        F2_state = s_F_state[(index+1)+nhorizon*fact_level]
#        indx_S=(index+1)+nhorizon*fact_level
        S = s_F_lambda[(index+1)+nhorizon*fact_level]

    # Perform dgemm operations
    S[:] = dgemm(alpha=1, a=C1_state, b=F1_state, beta=-1, c=S, trans_b=1)
    # print("S after first dgemm:\n", S)

    S[:] = dgemm(alpha=1, a=C1_input.T, b=F1_input, beta=1, c=S)
    # print("S after second dgemm:\n", S)
    S +=-1*F2_state
    # print("S final: ",indx_S, " \n")
    # print(S)
    
#Write tests
def getIndexFromLevel(nhorizon,depth,level,i,levels):
    num_nodes=np.power(2,depth-level-1)
    leaf=i*num_nodes//nhorizon
    count = 0
    for k in range (nhorizon):
        if(levels[k]!=level):
            continue
        if(count==leaf):
            return k
        count+=1
    return -1

def shouldCalcLambda(index, i,levels):
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
        lin_index=i+nhorizon*upper_level
        g_state = s_F_state[lin_index]
        g_input = s_F_input[lin_index]
        g_lambda = s_F_lambda[lin_index]


    F_state = s_F_state[i+nhorizon*level]
    F_input = s_F_input[i+nhorizon*level]
    F_lambda = s_F_lambda[i+nhorizon*level]

    if calc_lambda:
        g_lambda[:]=dgemm(alpha = -1,a=F_lambda, b=f,beta = 1,c=g_lambda)
    g_state[:]=dgemm(alpha=-1,a=F_state,b=f,beta=1,c=g_state)
    g_input[:]=dgemm(alpha=-1,a=F_input,b=f,beta=1,c=g_input)

