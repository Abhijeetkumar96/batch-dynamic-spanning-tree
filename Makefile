# Compiler settings
CC = g++
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
INCLUDE = -I$(CUDA_PATH)/include -Iinclude

# Compiler flags
CFLAGS = -std=c++11 -O3
NVCCFLAGS = -arch=sm_61 $(INCLUDE) # Adjust -arch based on your GPU

# Object files
OBJS = bfs.o cuda_utility.o dynamic_tree_util.o euler_v1.o list_ranking.o main.o

# Executable name
EXEC = my_program

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
