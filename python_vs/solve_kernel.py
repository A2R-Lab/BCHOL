import nested_dissect
import numpy as np
import math
import copy
import scipy.linalg as linalg
DEBUG = False

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

  if DEBUG:
     print(f"A matrix \n:{A[0]}")
     print(f"B matrix: \n{B[0]}")
     print(f"Q matrix \n:{Q[0]}")
     print(f"R matrix: \n{R[0]}")
     print(f"lambda matrix: \n{F_lambda[0]}")
     print(f"state matrix \n:{F_state[0]}")
     print(f"input matrix: \n{F_input[0]}")


     

  #make sure Q is not zero(add_epsln)
  #Q +=1e-5
  #SOLVE_LEAF is CORRECT
  for ind in range (knot_points):
      nested_dissect.solveLeaf(binary_tree,ind, state_size,knot_points,Q,R,q,r,A,B,d,F_lambda,F_state, F_input)

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
      num_perblock = num_factors//L
 

      #NEED TO FIX UPPER TRIANGLE VS LOWER TRIANGLE!
       #calc inner products Bbar and bbar (to solve y in Schur)
      for b_ind in range (L):
         for t_ind in range(cur_depth):
            ind = b_ind * cur_depth + t_ind
            leaf = ind // cur_depth
            upper_level = level + (ind % cur_depth)
            lin_ind = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
            nested_dissect.factorInnerProduct(A,B, F_state, F_input, F_lambda, lin_ind, upper_level, knot_points)

      #cholesky fact for Bbar/bbar
      for leaf in range (L):
         index = int(np.power(2.0, level)) * (2 * leaf + 1) - 1
         lin_ind = index + knot_points * level
         #check what you do with the pointer
         if(nested_dissect.is_choleskysafe(F_lambda[lin_ind+1])):
            F_lambda[lin_ind+1]=linalg.cho_factor(F_lambda[lin_ind+1],lower =True)[0]
         else:
            print(f"Can't factor Cholesky {lin_ind} :\n")
            print(F_lambda[lin_ind])

      

      #solve with Chol factor for y
      for b_id in range(L):
         for t_id in range(upper_levels):
            i = b_id*upper_levels+t_id
            leaf = i//upper_levels
            upper_level = level+1+(i%upper_levels)
            lin_ind = int(np.power(2,level)*(2*leaf+1))
            Sbar = F_lambda[(lin_ind)+knot_points*level]
            f = F_lambda[(lin_ind)+knot_points*upper_level]

            if(nested_dissect.is_choleskysafe(Sbar)):             
             f[:]=linalg.cho_solve((Sbar,True),f,overwrite_b=True)
            else:
               print("Cant sovle Chol")
               
      # print("after solveChol")
      # for i in range(knot_points*depth):
      #     print(f"F_lamda {i} \n: {F_lambda[i]}")
      #     print(f"F_state {i}:\n{F_state[i]}")
      #     print(f"F_input{i}: \n {F_input[i]}")
      

   # update SHUR - update x and z compliments      
      for b_id in range(L):
         for t_id in range(num_perblock):
            i = (b_id*4)+t_id
            k = i//upper_levels
            upper_level = level+1+(i%upper_levels)
            index = nested_dissect.getIndexFromLevel(knot_points,depth,level,k,binary_tree)
            calc_lambda  = nested_dissect.shouldCalcLambda(index, k,binary_tree)
            print(calc_lambda)

            print(f"norm i {i}, index {index}, calc lambda {calc_lambda}\n")
            g = k+knot_points*upper_level
            nested_dissect.updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,calc_lambda,knot_points)
      # print("after SHUR")
      # for i in range(knot_points*depth):
      #     print(f"F_lamda {i} \n: {F_lambda[i]}")
      #     print(f"F_state {i}:\n{F_state[i]}")
      #     print(f"F_input{i}: \n {F_input[i]}")


   #soln vector loop 
  for level in range (depth):
     L = int(np.power(2.0,(depth-level-1)))
     indx_atlevel = nested_dissect.getValuesAtLevel(binary_tree,level)
     count = len(indx_atlevel)
     num_perblock = knot_points // count
     print("started soln L ",L)
     print("level ",level)

     
   #calculate inner products with rhc
     for leaf in range(L):
         lin_ind = int(np.power(2,level)*(2*leaf+1)-1)
         nested_dissect.factorInnerProduct(A,B,q,r,d,lin_ind,0,knot_points,sol=True)
   
   #solve for separator vars with Cached cholesky
     for leaf in range(L):
         lin_ind = int(np.power(2,level)*(2*leaf+1)-1)
         Sbar = F_lambda[level * knot_points + (lin_ind + 1)]
         zy = d[lin_ind+1]
         if(nested_dissect.is_choleskysafe(zy)):    
            zy[:]=linalg.cho_solve((Sbar,True),zy,overwrite_b=True)
         else:
            print("Cant sovle Chol in soln")



      #propogate info to soln vector
     for b_id in range(L):
         for t_id in range(num_perblock):
            k = b_id * num_perblock + t_id
            index = nested_dissect.getIndexFromLevel(knot_points,depth,level,k,binary_tree)
            calc_lambda = nested_dissect.shouldCalcLambda(index,k,binary_tree)
            nested_dissect.updateShur(F_state,F_input,F_lambda,index,k,level,upper_level,
                                       calc_lambda,knot_points,sol=True,d=d,q=q,r=r)


  #need to double check results but the code runs          
  print("Done with rsLQR, soln:\n")
  for i in range(knot_points):
        print(f"d_{i} {d[i]}")
        print(f"q_{i}  {q[i]}")
        print(f"r_{i} {r[i]}")




        



     
     

     
     
   
      

           




