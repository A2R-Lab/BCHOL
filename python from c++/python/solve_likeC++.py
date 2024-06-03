#general imports
import copy
import sys
import math
import numpy as np
import scipy
import json
import csv
#import solve
#import solver

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
        soln = data['soln']
        print("nhorizon", nhorizon)
        print("x0", x0)


        # Initialize arrays for each variable
        Q_R_list = []
        q_r_list = []
        c_list = []
        A_B_list = []
        d_list = []
        d_list.append(x0)

        # Access elements inside lqrdata (assuming it's a list of dictionaries)
        for lqr in lqrdata:
            index = lqr['index']
            nstates = lqr['nstates']
            ninputs = lqr['ninputs']
            Q_R_list.append(lqr['Q'])
            Q_R_list.append(lqr['R'])
            q_r_list.append(lqr['q'])
            q_r_list.append(lqr['r'])
            c_list.append(lqr['c'])
            A_B_list.append(lqr['A'])
            A_B_list.append(lqr['B'])
            if(index!=nhorizon):
                d_list.append(lqr['d'])

    # Accessing solution data
    print("\nSolution data:")
    for sol in soln:
        print(sol)
    print("d ", d_list)
            
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
print("d",d_list)

    
