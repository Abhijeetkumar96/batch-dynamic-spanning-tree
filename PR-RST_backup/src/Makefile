# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++17 -Wall
NVCCFLAGS = -O2 -arch=sm_61  # Change the architecture flag based on your GPU
DEBUGFLAG =

# Source files
CPP_SOURCES = verifySol.cpp
CU_SOURCES = main.cu utility.cu rootedSpanningTreePR.cu grafting.cu reRoot.cu reversePaths.cu shortcutting.cu
OBJECTS = $(CPP_SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.o)

# Final target
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJECTS) -lcudart

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEBUGFLAG) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAG) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

debug: NVCCFLAGS += $(DEBUGFLAG)
debug: CXXFLAGS += $(DEBUGFLAG)
debug: all

.PHONY: all clean debug
