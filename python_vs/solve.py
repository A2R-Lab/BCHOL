#general imports

import pdb
import copy
import scipy
import sys
import math
import numpy as np
import json
import csv

#function specific imports
import nested_dissect
#import solve_kernel


#import solve

def runSolve():
    """
    runSolve is the main interface that lets you execute all the
    calculations to solve the LQR problem.
    """

# Prompt the user to select the file type
file_type = input("Enter 'json' or 'csv' to choose the file type: ")    

# Read data based on user's choice
if file_type == 'json':
    file_name = input("Enter the JSON file name: ")
    with open(file_name,'r') as file:
        data = json.load(file)
 
        # Access the top-level elements
        nhorizon = data['nhorizon']
        x0 = data['x0']
        lqrdata = data['lqrdata']
        soln = np.array(data['soln']).flatten()
        print("nhorizon", nhorizon)
        print("x0", x0)

        # Initialize arrays for each variable
        Q_list = []
        R_list = []
        q_list = []
        r_list = []
        c_list = []
        A_list = []
        B_list = []
        d_list = []
        d_list.append(x0)

        # Access elements inside lqrdata (assuming it's a list of dictionaries)
        for lqr in lqrdata:
            index = lqr['index']
            nstates =lqr['nstates']
            ninputs = lqr['ninputs']
            Q_list.append(lqr['Q'])
            R_list.append(lqr['R'])
            q_list.append(lqr['q'])
            r_list.append(lqr['r'])
            c_list.append(lqr['c'])
            A_list.append(lqr['A'])
            B_list.append(lqr['B'])
            if(index!=nhorizon):
                d_list.append(lqr['d'])

    # Accessing solution data
    print("\nSolution data:")
    for sol in soln:
        print(sol)
            
elif file_type == 'csv':
    file_name = input("Enter the CSV file name: ")
    with open(file_name,'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        #fix the csv example later
else:
    print("Invalid file type.")

#print the data
print("nhorizon",nhorizon)
print("nstates",nstates)
print("ninputs",ninputs)
print("A",A_list)
print("d",d_list)

#transform the lists to numpy arrays
Q =np.array([np.diag(row) for row in Q_list]) 
R = np.array([np.diag(row) for row in R_list])
q = np.array(q_list)
r = np.array(r_list)
A = np.array(A_list)
B_v = np.array(B_list)
B = np.zeros((nhorizon,nstates,ninputs)) 
#reshape B
for i in range(B.shape[0]):
    first_part = B_v[i,:,:ninputs]
    second_partt = B_v[i,:,ninputs:]
    B[i] = np.vstack((first_part,second_partt))

d = np.array(d_list)
c = np.array(c_list)
depth = int(math.log2(nhorizon))

nhorizon=int(nhorizon)
nstates= int(nstates)
ninputs=int(ninputs)


#A = A.reshape(-1,A.shape[-1]) #makes a long list of A matrices
#by this point we are done with data preparation, all the arrays are 3d 
#first dimension is the index

#imitating calling the kernel
#solve_kernel(nhorizon,ninputs,nstates,Q,R,q,r,A,B,d,F_lambda,F_state,F_input)






    
