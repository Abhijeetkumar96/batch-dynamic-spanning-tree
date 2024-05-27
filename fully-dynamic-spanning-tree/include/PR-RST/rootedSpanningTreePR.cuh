#ifndef ROOTED_SPANNING_TREE_PR_CUH
#define ROOTED_SPANNING_TREE_PR_CUH

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

int* RootedSpanningTree(int* d_u_arr, int* d_v_arr, const int numVert, const int numEdges);
// void RootedSpanningTree(int *d_u_ptr, int *d_v_ptr, const int n, const int edges, PR_RST& mem_mag);

#endif // ROOTED_SPANNING_TREE_PR_CUH