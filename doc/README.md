# Documentation

This directory contains detailed documentation for the project, including links to related papers, code variable explanations, implementation details and diagrams, and unique features of the CUDA-based implementation.
---


## Table of Contents
- [Paper Link](#paper-link)
- [Paper Overview](#paper-overview)
- [Code Dictionary for Variables](#code-dictionary-for-variables)
- [Code Overview](#code-overview)
- [Specific CUDA Code Overview](#specific-cuda-code-overview)
- [Implementation Uniqueness](#implementation-uniqueness)

---

## Paper Link

**[A Parallell Linear System Solver for Optimal Control](https://bjack205.github.io/papers/rslqr.pdf)** 

This documentation is designed to enhance the understanding of the algorithm and methodology outlined in the paper [A Parallel Linear System Solver for Optimal Control by Brian E. Jackson.

Specifically, it aims to:

  - Provide a clearer interpretation of the paper's concepts through additional examples and diagrams.
 - Serve as a terminology reference, bridging the naming conventions used by Brian and those used in the GATO repository.
 - Explain the unique aspects of the CUDA implementation, emphasizing its structure, efficiency, and parallelization approach.
   
Brian's original documentation can be accessed [here](https://github.com/bjack205/rsLQR/tree/main/docs).

---

## Paper Overview

The paper focuses on solving the **Linear Quadratic Regulator (LQR)** problem by rearranging and partitioning the original KKT matrix and solving it with recursive schur compliments. Key steps of the solver method include:

- Rearranging the KKT matrix

| Original KKT Matrix                                         | Rearranged KKT Matrix                                      |
|-------------------------------------------------------------|-----------------------------------------------------------|
| ![Original KKT Matrix](https://github.com/user-attachments/assets/96655878-40e9-4f9a-8be2-20f87b284b60) | ![Rearranged KKT Matrix](https://github.com/user-attachments/assets/58abfb9f-6d7d-4c06-b056-6d0e2f9ff6aa) |
| **Original KKT Matrix**                                     | **Rearranged KKT Matrix**                                 |



- Recursively Applying Schur Compliments to a Rearranged KKT Matrix

  
| Schur Compliments Matrix | Rearranged KKT Matrix |
|--------------------------|-----------------------|
| ![Original KKT Matrix](https://github.com/user-attachments/assets/f62e1adf-b8db-498a-bd3b-d06c58db73c4) | ![Rearranged KKT Matrix](https://github.com/user-attachments/assets/58abfb9f-6d7d-4c06-b056-6d0e2f9ff6aa) |
| **Schur Compliments Matrix** | **The red block in the upper left corner is A, the red lower-right block is C. The top vertical green block is D, and the bottom green vertical block is E.** |

  
- Solving with Schur from the lowest levels  - bottom up to the original KKT

| Lowest Partition (Each System is Independent) | First Level (Each System Consists of Two Sequential Timesteps) | Second Level (Original System of 4 Timesteps) |
|------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------|
| ![The lowest partition of the matrix - each system is independent](https://github.com/user-attachments/assets/591f1008-fbd0-4615-beac-13b258274b6d) | ![First level - each system is consisting of two sequential timesteps](https://github.com/user-attachments/assets/94b0eaac-8fae-451f-b94b-b9af20e13565) | ![Second level - the original system of 4 timesteps](https://github.com/user-attachments/assets/58abfb9f-6d7d-4c06-b056-6d0e2f9ff6aa) |



### Abstract Summary:
*Using a recursive application of
Schur compliments, the algorithm has a theoretical O(log(N))
complexity in the time horizon, and maps well onto many-core
processors. [cited from A Parallel Linear System Solver for Optimal Control]*

---

## Code Dictionary for Variables

Hence there are many different names for Control and State variables  we provide this table to further facilitate understanding of our code in BCHOL (corresponding to the paper vars) and its integration into the [GATO](https://github.com/A2R-Lab/GATO) repository. 

| Variable Name in the Paper    | Access in BCHOL        | Variable Name in GATO       | Description                              |
|-------------------------------|------------------------|-----------------------------|------------------------------------------|
| `A` - State-transition matrix.| `A_B` array            | `A` is part of the `C_dense`| The matrix that defines how the state evolves over time. |
| `B` - Control matrix.         | `A_B` array            | 'B' is part of the `C_dense`| The matrix that relates the control inputs to the system state. |
| `Q` - State cost matrix.      | `Q_R` array            | 'Q' is part of the `G_dense`| A matrix used in the cost function that penalizes deviations from the desired state. |
| `R`- Control cost matrix.     | `Q_R` array            | `R` is part of the `G_dense`| A matrix used in the cost function that penalizes control efforts. |
| `q` - state cost vector       | `q_r` array            | q is part of the `g_dense`  | A vector used in the cost function to penalize deviations in the system state. |
| `r`- control cost vector      | `q_r` array            | r is part of the `g_dense`  | A vector used in the cost function to penalize control efforts. |
| `d/f`- DOUBLE CHECK            | `d` array             | DOUBLE CHECK                | External disturbance or offset in the system dynamics. |
| `x` - system state vector     | the system solves in place and puts x values into `q_r` instead of the q vector | `x` is part of the dxul   | A vector that represents the state of the system at a given time. |
| `u` - control input vector   | the system solves in place and puts u values into `q_r` instead of the r vector| `u` is part of the dxul  | A vector that represents the control input applied to the system. |
| `λ` (Lambda) - dual variables     |  the system solves in place and puts lambda values into `d`  | lambda is part of the `dxul`  | A vector represents Lagrange multipliers or dual variables used to enforce constraints in optimization problems


*It's important to mention that all matrices are column-major order and B matrix is transformed in BCHOL examples and BCHOL's csv files.


---

## Files overview

The code is structured into the following main components:

1. **Solver launch (`solve_lqr.cu`)**:  
   The initialization of LQR problem, memory allocation and launch of CUDA kernel that implements the LQR solution.

2. **The main Kernel (`solve.cuh`)**:  
   Contains the main BCHOL kernel and calls to the helper functions.
   
4. **../help_functions**:  
   Declaration and implementation of all the smaller functions that build up the algorithm
   
   4.1 `add_epsln.cuh` - adding an epsilon to ensure that matrix is positive definit
   
   4.2 `chol_InPlace.cuh` - performs Cholesky Factorization in place (plan to substitute with GLASS function)

   4.3 `copy_mult.cuh` - modified copy functions of several arrays

   4.4 `csv.cuh `- a helper funciton to read and write csv examples

   4.5 `diag_Matrix_set.cuh` - sets the diagonal of a matrix to a specific number (plan to call GLASS instead)

   4.6 `lowerBackSub.cuh` - performs lower back substitution (plan to call GLASS instead in the future)

   4.7 `nested_dissect.cuh` - the main file of the smaller functions for the algorithm (solve_leaf ; factorInnerProduct ; shouldCalcLambda ; updateSchur)

   4.8 `print_debug.cuh` - a helper print function for customized debugging

   4.9 `scaled_sum.cuh` - computes the scaled sum of two matrices (plan to call GLASS instead in the future)

   4.10 `set_const.cuh` - sets the whole vector/matrix to a constant (plan to call GLASS instead)

   4.11 `tree_functs.cuh` - functions that build and return values from the binary tree

6. **Makefile**:  
   Automates the build process for compiling the CUDA and host code.

7. **Tests (`tests/unit_tests`)**:  
   Includes unit tests for validating key components of the implementation.

---
## Memory Design

Before we dive into our solver method it's important to understand the underlying memory structure. We will quickly inspect the original memory design in the C- implementation, then switch to our CUDA memory design and explain the details about the imports between shared and device memory. 

![Memory Layout](https://github.com/user-attachments/assets/b6771ba9-9d61-405c-aa21-e472f816521d)

### NDData 
[NDdata](https://github.com/bjack205/rsLQR/blob/main/src/nddata.h)  helps us to separate: 

**A)** Initial Dynamic Matrices inside ND Data_Data**

**B)** Factorized matrices that will hold solutions to intermediate levels inside ND Data_Fact 

**C)** soln vector inside ND Data_soln (initially b vector that is solved in place)


### NDFactor 

NDfactor is the underlying structure that holds a chunk of memory for a single time step. It stores it in a way of a matrix size(2n+m) divided into blocks:

\[
\begin{bmatrix} 
\Lambda \\ 
X \\ 
U 
\end{bmatrix}
\]

### Binary Tree Structure
Last but not least, let's talk about the tree structure. After we refactorzied our [matrix](https://github.com/A2R-Lab/BCHOL/tree/main/doc#:~:text=Solving%20with%20Schur%20from%20the%20lowest%20levels%20%2D%20bottom%20up%20to%20the%20original%20KKT) and solved the independent equations we have two levels of systems for horizon 4. We can see that the matrix structure resembles the binary tree with *-I* matrix being next to the actual time step. The Binary tree structure is the underlying structure of ND Data_Data and ND Data_Fact.

*Examples :*
|Binary Tree for N=4| ND Data | 
|----------|----------|
| ![image](https://github.com/user-attachments/assets/60807e1b-3174-4aff-aa9e-7c45b1bda484) | ![image](https://github.com/user-attachments/assets/ba2d6781-2f38-4ec7-adae-b08b691c4b42) | 

| Binary Tree for N=8 | ND Data |
|----------------------|---------|
| ![image](https://github.com/user-attachments/assets/30a611ef-6d6f-4681-9d38-f4083b9fe147) | ![image](https://github.com/user-attachments/assets/93f11632-2e97-44fe-b5f9-fcb5b078161b) |

### CUDA Implementation

For the CUDA Implementation we are abondoning the ND_Data and ND_Factor structures. 

* ND Data_Data which contains original A and B matrices instead is saved into A_B 1D array, where A and B.T are saved in column-major order and sequentually stored (A1, B1.T, A2,B2.T...)
*Q,R matrices are saved in condensed state (only diagonals) in 1D array sequentially (Q1,R1,Q2,R2..)
*q,r vectors are saved in the same manner in q_r array
*d vector is saved as a separate d array.

It is important to emphasize that q,r,d are solved in place and become x,u,lambda - solution vector.

Finally for the ND Data_Fact (the factorized matrices that keep the intermediate solutions of matrices and Cholesky factorizations) the tree structure is very crutial as the logic of the algorithm is based on the levels. Hence we are declaring 3 different arrays

* **F_lambda** - an array of all the lambdas from the ND Factors stores sequentally level by level, for example for n=8 it'll be of size 24(8 timesteps*3 levels)×state×state
* **F_state** - an array of all the states from the ND Factors stores sequentyally level by level
* * **F_input** - an array of all the states from the ND Factors stores sequentyally level by level
  
# Specific CUDA Code Overview
[Include diagrams and explain the F-factor; F_lambda and etc../]
The CUDA kernel implementation in `solve_lqr.cu` is designed to leverage the parallel processing power of NVIDIA GPUs. Key features include:

- **Thread-Level Parallelism**:  
  Each thread computes part of the solution for a state/control update.

- **Shared Memory Optimization**:  
  Frequently accessed data (e.g., matrices `A`, `B`, `Q`, `R`) is stored in shared memory to reduce global memory access overhead.

- **Memory Coalescing**:  
  Data access is optimized for efficient coalescing to minimize latency.

- **Use of cuBLAS**:  
  The `cuBLAS` library is employed for matrix operations like multiplication, ensuring high-performance computation.

<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
   <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
   Sorry, your browser does not support inline SVG.
</svg> 
---

## Implementation Uniqueness

The implementation distinguishes itself with the following unique features:

1. **Custom CUDA Kernel for Parallel Matrix Updates**:  
   - Unlike typical cuBLAS-only approaches, this implementation includes custom kernels for parts of the computation, enabling fine-grained control over memory and execution flow.

2. **Low Latency for Real-Time Applications**:  
   - Specifically optimized for scenarios where rapid LQR computations are required, such as robotics and control systems.

3. **Scalability**:  
   - The code scales efficiently with problem size and GPU resources, leveraging hierarchical parallelism (blocks and threads).

4. **Error Debugging with Device Flags**:  
   - The `-g -G` flags in the `Makefile` enable device-side debugging for easier identification of runtime issues.

---

*For further details, consult the source code and accompanying paper linked above.*
