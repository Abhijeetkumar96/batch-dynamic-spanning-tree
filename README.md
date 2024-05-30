# CUDA-based Fully Dynamic Spanning Tree

## Introduction

This repository contains a CUDA-based multicore solution designed to repair a spanning tree after a batch of edges is deleted. Using parallel processing on a GPU, this algorithm ensures that the spanning tree remains connected even after certain edges have been removed.

## Features

- **GPU processing**: Harnesses the power of the GPU to handle multiple tasks simultaneously.
- **Batched edge removal**: Allows a batch of edges to be inserted/deleted at once and repairs the spanning tree accordingly.
- **High performance**: Designed to leverage CUDA for efficient computations.

## Prerequisites

- NVIDIA GPU (with CUDA support)
- CUDA Toolkit and Compatible Compiler (e.g., nvcc)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhijeetkumar96/batch-dynamic-spanning-tree
   ```
2. **Navigate to the cloned directory:**
   ```bash
   cd fully-dynamic-spanning-tree
   ```
3. **Compile the code:**
  - **Build without the `DEBUGFLAG`:**
  ```bash
    mkdir build && cd build
    cmake ..
  ```
  ```bash
    make -j$(nproc)
  ```

**Remember to run make clean in between if you switch build types, to ensure that all objects and targets are properly rebuilt.**

## Usage

**Prepare your input files**: This should include:
- A file describing the graph G (e.g., `graph.txt`).
- A file containing a batch of edges to delete or insert (e.g., `edges_to_delete.txt` or `edges_to_insert.txt`).

**Run the program**:

```bash
  ./main -i graph.txt -b edges_to_delete.txt -r HS -p ET -d 0
```
## Input Format

### Graph File (`graph.txt`)

```bash
  nodes vertices
  node1 node2
  node2 node3
  ...
```

### Edges to Insert/Delete File (`edges_to_delete.txt`)

The edges to delete or insert should also be described with one edge per line:

```bash
total number of edges to delete/ insert
node1 node2
node2 node3
...
```

## Contributing

If you'd like to contribute, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

Unlicensed

  


