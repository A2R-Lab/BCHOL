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
  
<div style="display: flex; flex-direction: row; align-items: center; justify-content: space-around;">

  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/96655878-40e9-4f9a-8be2-20f87b284b60" alt="Original KKT matrix" width="300" />
    <p>Original KKT Matrix</p>
  </div>

  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/58abfb9f-6d7d-4c06-b056-6d0e2f9ff6aa" alt="Rearranged KKT matrix" width="300" />
    <p>Rearranged KKT Matrix</p>
  </div>

</div>


- Performance comparisons with CPU-based and other GPU-based methods.
- Application to real-time systems where high throughput and low latency are critical.

### Abstract Summary:
*Insert the abstract or a summarized version here, highlighting the core contributions.*

---

## Code Dictionary for Variables

| Variable Name | Description                                     |
|---------------|-------------------------------------------------|
| `A`           | State-transition matrix.                       |
| `B`           | Control matrix.                                |
| `Q`           | State cost matrix.                             |
| `R`           | Control cost matrix.                           |
| `K`           | Optimal feedback gain matrix.                  |
| `X`           | System state vector.                           |
| `U`           | Control input vector.                          |

*Note:* Include additional variables and their descriptions as needed.

---

## Code Overview

The code is structured into the following main components:

1. **Main Solver (`solve_lqr.cu`)**:  
   The CUDA kernel that implements the LQR solution.

2. **Header File (`solve.cuh`)**:  
   Contains function declarations, constants, and data structures.

3. **Makefile**:  
   Automates the build process for compiling the CUDA and host code.

4. **Tests (`tests/unit_tests`)**:  
   Includes unit tests for validating key components of the implementation.

---

## Specific CUDA Code Overview

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
