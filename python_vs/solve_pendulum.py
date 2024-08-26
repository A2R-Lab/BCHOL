#general imports

import pdb
import copy
import scipy
import sys
import math
import numpy as np
import json
import csv
INIT = False

#function specific imports
import nested_dissect
import solve_kernel


#check if they are lists or np.arrays
def solve_Pendulum(G: np.ndarray, g: np.ndarray, C: np.ndarray, c: np.ndarray, N: int):
    """
    Prepares the matrices to the right format in order to launch LQR kerne;

    Parameters:
    G (np.ndarray): A matrix of Q_R from pendulum problem .
    g (np.ndarray): A combined vector of q_r
    C (np.ndarray): A matrix of A_B and I, negated
    c (np.ndarray): A d vector

    Returns:
    CHECK with Brian what it needs to return
    """    """
    runSolve is the main interface that lets you execute all the
    calculations to solve the LQR problem.
    """
    

    nhorizon = N
    # nhorizon = 8
    nstates = 2
    ninputs =1


    #Preparing Q as a separate array 
    #Q must be in shape (nhorzon,nstates,nstates) == (knot_points,state_size,state_size)

    Q =np.diag([3.001,4.001])
    QF = np.diag([100.001,100.001])    
    Q_stacked = np.stack([Q] * (nhorizon - 1))
    # Add the final matrix QF to the end of the stacked array
    Q= np.concatenate([Q_stacked, QF[np.newaxis, :, :]])

    #Preparing R as a separate array
    #R must be in shape (nhorzon,ninputs,ninputs) == (knot_points,ninputs,ninputs)
    R =np.diag([0.101])
    R= np.stack([R] * (nhorizon - 1))
    #add 0s for last timestep
    R=np.concatenate([R,np.zeros((1,1,1))])

    #preparing q_r as separate vector, add r=0 at the last timestep
    g=np.append(g,np.zeros(ninputs))

    #q must be in shape (nhorzon,nstates) == (knot_points,state_size)
    #extract q from g
    g_reshaped = g.reshape(-1, 3)
    q = g_reshaped[:, :2].flatten()
    q =q.reshape(-1,2)
    #extract r from g
    r = g_reshaped[:,-1].flatten()
    r=r.reshape(-1,1)

    #get A,B from C
    A_list =[] 
    B_list =[]
    for i in range (nhorizon-1):
            row = nstates+i*nstates
            col =i*(nstates+ninputs)
            A_temp = C[row:row+nstates,col:col+nstates]
            B_temp = C[row:row+nstates,col+nstates]
            A_list.append(A_temp)
            B_list.append(B_temp)
    A = np.array(A_list) 
    B = np.array(B_list)
    B=B.reshape(7,2,1)
    #transpose B
    B = B.transpose(0, 2, 1)
    #add 0s at the last timestep
    A = np.concatenate(A,np.zeros(1,2,2))
    B=np.concatenate([B,np.zeros((1,2,1))])

    #negate both A and B
    A=-A
    B=-B

    #get d (just copy c vector)
    d=c[:]
    d=d.reshape(-1,2)

    depth = int(math.log2(nhorizon))


    #INIT looks identical to the solve_lqr.cuh, 3D array use A[i] to get the matrix
    if(INIT):
        for i in range(nhorizon):
            print(f"A matrix \n:{i} , {A[i]}")
            print(f"B matrix: \n{B[i]}")
            print(f"Q matrix \n:{Q[i]}")
            print(f"R matrix: \n{R[i]}")
            print(f"q  {q[i]}")
            print(f"r {r[i]}")
            print(f"d {d[i]}")



    #imitating calling the kernel
    solve_kernel.solve_kernel(nhorizon,ninputs,nstates,Q,R,q,r,A,B,d)





