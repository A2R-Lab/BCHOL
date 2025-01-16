# BCHOL - rsLQR implementation in CUDA

Contains the CUDA accelerated code of the Recursive Schur LQR algorithm. The original C implementation can be found here: https://github.com/bjack205/rsLQR/

Python implementation of the code can be found [here](https://github.com/A2R-Lab/BCHOL-python)

## Table of Contents  
1. [Introduction](#introduction)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Usage](#usage)
5. [Documentation](#documentation)
6. [Citing](#citing)  

## Introduction
Solves for x in Ax = b, using the Recursive Schur Linear Quadratic Regulator explained in the paper [A Parallell Linear System Solver for Optimal Control](https://bjack205.github.io/papers/rslqr.pdf) by Brian E.Jackson. It requires A to be a positive semi-definite matrix to guarantee a good result.

This method is part of the GATO solver (GPU Accelerated Trajectory Optimization) algorithms and model predictive control (MPC). Learn more about  GATO [here](https://github.com/A2R-Lab/TrajoptMPCReference).


## Requirments

- **NVIDIA GPU** with Compute Capability **8.6** or higher (e.g., NVIDIA Ampere-based GPUs like A100, RTX 30 series).

1. **CUDA Toolkit**:
   - Version **11.4** or higher.
   - Includes `nvcc` compiler and runtime libraries.
   - Download: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

2. **C++ Compiler**:
   - **g++ 9.4** (for Linux) or a compatible version.

3. **Build Tools**:
   - `make` utility is required to build the project.
  
## Installation

   This solver uses our GPU Linear Algebra Simple Subrotoutines (GLASS) library as a submodule, as it has proved to be faster for smaller problems than the traditional cuBLASS library.
   
   Clone the repository and initialize the submodule, then run *make* command in the src folder.
   ```bash
   git clone  --recurse-submodules https://github.com/A2R-Lab/BCHOL.git
   cd BCHOL
   cd src
   make
```

## Usage
After running the make command you should have the *./rsLQR* executable in the src folder.  Run it to see the [example](https://github.com/bjack205/rsLQR/blob/main/lqr_prob.json) of the solver provided by Brian Jackson. You can run your example by providing 'json' file of the LQR problem. Check *solve_lqe.cu* for the example of usage.

`./rsLQR`

## Documentation
It is highly recommended to read the paper [A Parallell Linear System Solver for Optimal Control](https://bjack205.github.io/papers/rslqr.pdf) by Brian E. Jackson prior to using this code.

Additional documentation and graphical explanation has been provided by Yana Botvinnik (here)[]


