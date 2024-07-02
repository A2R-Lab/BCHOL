# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

# Google Test directory:
GTEST_DIR=/usr/local/gtest

# CC compiler options:
CC=g++
CC_FLAGS=-O3  -I$(GTEST_DIR)/include
CC_LIBS=-L$(GTEST_DIR)/lib -lgtest -lgtest_main -lpthread

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-O3 --compiler-bindir=/usr/bin/g++-11
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR=-L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR=-I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS=-lcudart

## Compile ##

OBJS = test_main.o

test_main: $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CC_LIBS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

test_main.o: test_main.cu test_chol.cuh test_copy.cuh test_matrix.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) $(CUDA_INC_DIR)