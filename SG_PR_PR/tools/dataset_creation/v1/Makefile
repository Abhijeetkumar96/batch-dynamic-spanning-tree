# Compiler to use
CXX=g++

# Compiler flags
CXXFLAGS=-std=c++17 -O3

# Name of the executable
EXEC=dataset_creation

# Source file
SOURCE=main.cxx

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): main.cxx undirected_graph.hxx dataset_creation.hxx
	$(CXX) $(CXXFLAGS) -o $@ $<

# Cleaning up
clean:
	rm -f $(EXEC)

# Phony targets
.PHONY: all clean
