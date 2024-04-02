# Compiler settings
CC = g++
NVCC = nvcc
CUDA_PATH = /usr/local/cuda

# Include and Library Paths
INCLUDES = -I./include -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Compiler flags
CFLAGS = -std=c++11 -O3
NVCCFLAGS = -arch=sm_61 -O3 $(INCLUDES) # Adjust -arch based on your GPU

# Source and Object files
CPP_SOURCES = $(wildcard src/*.cpp)
CU_SOURCES = $(wildcard src/*.cu)

CPP_OBJS = $(CPP_SOURCES:src/%.cpp=obj/%.o)
CU_OBJS = $(CU_SOURCES:src/%.cu=obj/%.o)
OBJS = $(CPP_OBJS) $(CU_OBJS)

# Output binary
EXEC = spanning_tree

# Default make target
all: obj $(EXEC)

obj:
	mkdir -p obj

$(EXEC): $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

obj/%.o: src/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

obj/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean objects in the obj directory and the executable
clean:
	rm -rf obj $(EXEC)
