import nested_dissect
import numpy as np
import math
import copy
import scipy.linalg as linalg

def solve_kernel(knot_points,control_size, state_size,
                  Q,R,q,r,A,B,d,F_lambda,F_state,F_input):
  #KKT constants
  states_sq = state_size * state_size
  inputs_sq = control_size*control_size
  inp_states = control_size*state_size
  depth = int(np.log2(knot_points))
  binary_tree = nested_dissect.initBTlevel(knot_points)


  #negate q_r and d vectors
  q=-q
  r = -r
  d = -d

  #Set F_lambda,F_state, and F_input
  F_lambda = np.zeros((knot_points*depth,state_size,state_size))
  F_state = np.zeros((knot_points*depth,state_size,state_size))
  F_input = np.zeros((knot_points*depth,state_size,control_size))


  #maybe immitate copying over here like in kernel?


  #SOLVE_LEAF
  for ind in range (knot_points):
    nested_dissect.solveLeaf(binary_tree,ind, state_size,control_size,Q,R,q,r,A,B,d,F_lambda,F_state,F_input)
    #imitate copying here to RAM later

    #update *shared memory*

    #Starting big loop
    for level in range (depth):
        #get the vars for the big loop
        count = #get values at level - BUILD DICTIONARY OR A MAP!
        L = math.pow(2.0,(depth-level-1))
        cur_depth = depth-level
        upper_levels = cur_depth-1
        num_factors = knot_points*upper_levels
        num_perblock = num_factors/L
        #lots of copying again between ram and shared

        #calc inner products Bbarand bbar (to solve y in Schur)
        for b_ind in range (L):
           for t_ind in range(cur_depth):
              ind = ind = b_ind * cur_depth + t_ind
              leaf = ind / cur_depth
              upper_level = level + (ind % cur_depth)
              lin_ind = pow(2.0, level) * (2 * leaf + 1) - 1
              nested_dissect.factorInnerProduct<float>(A,B, F_state, F_input, F_lambda, lin_ind, upper_level, state_size, control_size, knot_points)

        #cholesky fact for Bbar/bbar
        for leaf in range (L):
           index = pow(2.0, level) * (2 * leaf + 1) - 1
           lin_ind = index + nhorizon * level
           #check what you do with the pointer
           S = F_lambda + (states_sq * (lin_ind + 1))
           #substitue with linalg function
           nested_dissect.chol_InPlace<float>(state_size, S, cgrps::this_thread_block())
        
        #solve with Chol factor for y

      

