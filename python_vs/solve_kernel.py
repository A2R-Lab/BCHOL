import nested_dissect
import numpy as np
import math
import copy
import scipy.linalg as linalg

def solve_kernel(knot_points,control_size, state_size,
                  Q,R,q,r,A,B,d):
  #KKT constants
  states_sq = state_size * state_size
  inputs_sq = control_size*control_size
  inp_states = control_size*state_size
  depth = int(np.log2(knot_points))
  binary_tree = nested_dissect.initBTlevel(knot_points)
  print("Inside the kernel\n")

  #negate q_r and d vectors
  q=-q
  r = -r
  d = -d

  #Set F_lambda,F_state, and F_input
  F_lambda = np.zeros((knot_points*depth,state_size,state_size))
  F_state = np.zeros((knot_points*depth,state_size,state_size))
  F_input = np.zeros((knot_points*depth,control_size,state_size))

  #make sure Q is not zero(add_epsln)
  #Q +=1e-5
  #SOLVE_LEAF
  for ind in range (knot_points):
      nested_dissect.solveLeaf(binary_tree,ind, state_size,knot_points,Q,R,q,r,A,B,d,F_lambda,F_state, F_input)

   #imitate copying here to RAM later
  #update *shared memory*

   #Starting big loop
  for level in range (depth):
        #get the vars for the big loop
      print("started big loop")
      indx_atlevel = nested_dissect.getValuesAtLevel(binary_tree,level)

      count =len(indx_atlevel) 
      L = int(np.power(2.0,(depth-level-1)))
      print("L",L)
      cur_depth = depth-level
      upper_levels = cur_depth-1
      num_factors = knot_points*upper_levels
      num_perblock = num_factors/L
      print(count)

        #NEED TO FIX UPPER TRIANGLE VS LOWER TRIANGLE!
        #calc inner products Bbar and bbar (to solve y in Schur)
      for b_ind in range (L):
         for t_ind in range(cur_depth):
            ind = ind = b_ind * cur_depth + t_ind
            leaf = ind // cur_depth
            upper_level = level + (ind % cur_depth)
            lin_ind = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
            nested_dissect.factorInnerProduct(A,B, F_state, F_input, F_lambda, lin_ind, upper_level, knot_points)

        #cholesky fact for Bbar/bbar
      for leaf in range (L):
         index = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
         lin_ind = index + knot_points * level
         #check what you do with the pointer
         if(nested_dissect.is_choleskysafe(F_lambda[lin_ind])):
            F_lambda[lin_ind]=linalg.cho_factor(F_lambda[lin_ind],lower =True)[0]
         else:
            print(f"NOT PD {lin_ind} :\n")
            print(F_lambda[lin_ind])
                 
      #solve with Chol factor for y
      for b_id in range(L):
         for t_id in range(upper_levels):
            i = b_id*upper_levels+t_id
            leaf = i//upper_levels
            upper_level = level+1+(i%upper_level)
            lin_ind = int(np.power(2,level)*(2*leaf+1)-1)
            Sbar = F_lambda[(lin_ind+1)+knot_points*level]
            f = F_lambda[(lin_ind+1)+knot_points*upper_level]
            if(nested_dissect.is_choleskysafe(Sbar)):
             f[:]=linalg.cho_solve((Sbar,True),f,overwrite_b=True)
            else:
               print("Cant sovle Chol")

         #update Schur
      for b_id in range(L):
         for t_id in range(num_perblock):
            i = (b_id*4)+t_id
            k = i//upper_levels
            upper_level = level+1+(i%upper_levels)
            index = nested_dissect.getIndexFromLevel(knot_points,depth,level,k,binary_tree)
            calc_lambda  = nested_dissect.shouldCalcLambda(index, k ,knot_points,binary_tree)
            g = k+knot_points*upper_level
            nested_dissect.updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,calc_lambda,state_size,control_size,knot_points)

   #DONE WITH THE BIG LOOP
   #soln vector loop
   for level in range (depth):
   
      

           




