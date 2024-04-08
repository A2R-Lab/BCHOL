# Target executable
TARGET = chol_test

# Source files
SRCS = main.cu test_chol_InPlace.cuh test_chol.cuh

# Object files
OBJS = $(SRCS:.cu=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)