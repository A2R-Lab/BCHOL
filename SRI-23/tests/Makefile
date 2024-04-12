CXX := nvcc
CXXFLAGS := -std=c++11 $(CUDA_FLAGS)

all: chol_test

main.o: main.cu test_chol_InPlace.cuh test_chol.cuh
	$(CXX) $(CXXFLAGS) -c main.cu -o main.o

chol_test: main.o
	$(CXX) $(CXXFLAGS) main.o -o test

clean:
	rm -f main.o test