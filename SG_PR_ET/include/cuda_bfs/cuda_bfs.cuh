#ifndef CUDA_BFS
#define CUDA_BFS

#include "common/graph.hpp"

int* cuda_BFS(int* original_u, int* original_v, int numVert, long numEdges);

#endif // CUDA_BFS