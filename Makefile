# Specify the compiler
CC=nvcc
CXX=g++
# Specify compiler flags
CFLAGS=-std=c++17 -O3

# Target executable name
TARGET=euler

# Object files
OBJS=euler.o list_ranking.o main.o bfs.o

# Link object files into the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Compile each source file into an object file
euler.o: euler.cu
	$(CC) $(CFLAGS) -c euler.cu

list_ranking.o: list_ranking.cu list_ranking.cuh
	$(CC) $(CFLAGS) -c list_ranking.cu

bfs.o: bfs.cpp bfs.h
	$(CXX) $(CFLAGS) -c bfs.cpp

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu

# Clean target for removing compiled products
clean:
	rm -f $(TARGET) $(OBJS)
