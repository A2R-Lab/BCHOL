# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

## Compile ##

test: solve_lqr.o

solve_lqr.o: solve_lqr.cu solve.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

