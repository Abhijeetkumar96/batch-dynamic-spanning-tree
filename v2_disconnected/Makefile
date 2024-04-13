CC=g++
NVCC=nvcc
CFLAGS=-O3 -Wall -Iinclude
NVFLAGS=-arch=sm_61 -O3 -Iinclude

# Name of your executable
EXEC=eulerian_tour

# Directory for source files
SRC_DIR=src

# Directory for object files
OBJ_DIR=obj

# Ensure the object directory exists
$(shell mkdir -p $(OBJ_DIR))

# Add all cpp files you want to compile
CPP_SOURCES=$(addprefix $(SRC_DIR)/, spanning_tree.cpp)

# Add all cu files you want to compile
CU_SOURCES=$(addprefix $(SRC_DIR)/, main.cu euler.cu listranking.cu pointerJumping.cu)

# Generate object file names from sources
CPP_OBJECTS=$(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJECTS=$(CU_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Default rule
all: $(EXEC)

# Linking all objects to executable
$(EXEC): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ -L/usr/local/cuda/lib64 -lcudart

# Compiling CPP files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Compiling CU files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Cleaning up
clean:
	rm -f $(EXEC) $(CPP_OBJECTS) $(CU_OBJECTS)
	rm -rf $(OBJ_DIR)  # Optionally remove the obj directory

# Phony targets
.PHONY: all clean
