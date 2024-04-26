# Compiler settings - Can change cc to gcc or g++ if required
CC = g++
# Compiler flags, e.g. -g for debugging, -O2 for release
CFLAGS = -Wall -Iinclude -std=c++17

# Define the target executable
TARGET = dataset_creation

# Define object files
OBJ = obj/bridges.o obj/main.o

# Linking all object files to executable
$(TARGET): $(OBJ) include/*
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET)

# Compiling source files to object files
obj/%.o: src/%.cxx
	$(CC) -c $(CFLAGS) $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJ)

# Declare 'clean' as phony target
.PHONY: clean
