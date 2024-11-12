# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

# CC compiler options:
CC=g++
CC_FLAGS=-O3 
CC_LIBS=

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-g -G -O3  -gencode arch=compute_86,code=sm_86
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

## Compile ##

OBJS = solve_lqr.o

rsLQR: $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

solve_lqr.o: solve_lqr.cu solve.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean target
clean:
	rm -f $(OBJS) $(TARGET)
