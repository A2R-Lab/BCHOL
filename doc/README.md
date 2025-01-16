# Documentation

This directory contains detailed documentation for the project, including links to related papers, code variable explanations, implementation details, and unique features of the CUDA-based implementation.

---
<svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">
  <!-- Nodes -->
  <circle cx="50" cy="250" r="20" fill="lightblue" stroke="black" />
  <text x="45" y="255" fill="black">0</text>
  <circle cx="150" cy="250" r="20" fill="lightblue" stroke="black" />
  <text x="145" y="255" fill="black">2</text>
  <circle cx="250" cy="250" r="20" fill="lightblue" stroke="black" />
  <text x="245" y="255" fill="black">4</text>
  <circle cx="350" cy="250" r="20" fill="lightblue" stroke="black" />
  <text x="345" y="255" fill="black">6</text>
  <circle cx="100" cy="150" r="20" fill="lightgreen" stroke="black" />
  <text x="95" y="155" fill="black">1</text>
  <circle cx="300" cy="150" r="20" fill="lightgreen" stroke="black" />
  <text x="295" y="155" fill="black">5</text>
  <circle cx="200" cy="50" r="20" fill="pink" stroke="black" />
  <text x="195" y="55" fill="black">3</text>
  <!-- Edges -->
  <line x1="50" y1="230" x2="100" y2="170" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="0s" dur="1s" fill="freeze" />
  </line>
  <line x1="150" y1="230" x2="100" y2="170" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="1s" dur="1s" fill="freeze" />
  </line>
  <line x1="250" y1="230" x2="300" y2="170" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="2s" dur="1s" fill="freeze" />
  </line>
  <line x1="350" y1="230" x2="300" y2="170" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="3s" dur="1s" fill="freeze" />
  </line>
  <line x1="100" y1="130" x2="200" y2="70" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="4s" dur="1s" fill="freeze" />
  </line>
  <line x1="300" y1="130" x2="200" y2="70" stroke="black" stroke-width="2">
    <animate attributeName="stroke" from="gray" to="red" begin="5s" dur="1s" fill="freeze" />
  </line>
</svg>


## Table of Contents
- [Paper Link](#paper-link)
- [Paper Overview](#paper-overview)
- [Code Dictionary for Variables](#code-dictionary-for-variables)
- [Code Overview](#code-overview)
- [Specific CUDA Code Overview](#specific-cuda-code-overview)
- [Implementation Uniqueness](#implementation-uniqueness)

---

## Paper Link

The algorithm and methodology for this project are based on the following paper:

**[Insert Paper Title Here](https://example.com)**  
*Author(s):* [Author Names]  
*Published in:* [Conference/Journal Name]  

---

## Paper Overview

The paper focuses on solving the **Linear Quadratic Regulator (LQR)** problem using GPU-accelerated approaches. Key contributions of the paper include:

- A novel approach to solving LQR problems efficiently on NVIDIA GPUs.
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
