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
| `d/f`- DOUBLE CHECK            | `d` array             | DOUBLE CHECK                | A vector that saved the initial position and future [] |
| `x` - system state vector     | the system solves in place and puts x values into `q_r` instead of the q vector | 'x' is part of the dxul   | A vector that represents the state of the system at a given time. |
| `u` - control input vector   | the system solves in place and puts u values into `q_r` instead of the r vector| `u` is part of the dxul  | A vector that represents the control input applied to the system. |
| `λ` (Lambda) - dual variables     |  the system solves in place and puts lambda values into `d`  | lambda is part of the `dxul`  | A vector represents Lagrange multipliers or dual variables used to enforce constraints in optimization problems


*It's important to mention that B matrix is transformed in BCHOL 
** Maybe add F_lambda, F_state, F_input over here(??)

---

## Code Overview

The code is structured into the following main components:

1. **Solver launch (`solve_lqr.cu`)**:  
   The initialization of LQR problem, memory allocation and launch of CUDA kernel that implements the LQR solution.

2. **The main Kernel (`solve.cuh`)**:  
   Contains the main BCHOL kernel and calls to the helper functions.
   
4. **../help_functions**:  
   Declaration and implementation of all the smaller functions that build up the algorithm
   
   4.1 add_epsln.cuh - adding an epsilon to ensure that matrix is positive definit
   
   4.2 chol_InPlace.cuh - performs Cholesky Factorization in place (plan to substitute with GLASS function)

   4.3 copy_mult.cuh - modified copy functions of several arrays

   4.4 csv.cuh - a helper funciton to read and write csv examples

   4.5 diag_Matrix_set.cuh - sets the diagonal of a matrix to a specific number (plan to call GLASS instead)

   4.6 lowerBackSub.cuh - performs lower back substitution (plan to call GLASS instead in the future)

   4.7 nested_dissect.cuh - the main file of the smaller functions for the algorithm (solve_leaf ; factorInnerProduct ; shouldCalcLambda ; updateSchur)

   4.8 print_debug.cuh - a helper print function for customized debugging

   4.9 scaled_sum.cuh - computes the scaled sum of two matrices (plan to call GLASS instead in the future)

   4.10 set_const.cuh - sets the whole vector/matrix to a constant (plan to call GLASS instead)

   4.11 tree_functs.cuh - functions that build and return values from the binary tree

6. **Makefile**:  
   Automates the build process for compiling the CUDA and host code.

7. **Tests (`tests/unit_tests`)**:  
   Includes unit tests for validating key components of the implementation.

---

## Specific CUDA Code Overview
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
