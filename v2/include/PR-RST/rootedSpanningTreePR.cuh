#ifndef ROOTED_SPANNING_TREE_PR_CUH
#define ROOTED_SPANNING_TREE_PR_CUH

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

#include "PR-RST/pr_rst_util.cuh"

void RootedSpanningTree(int* d_edge_u, int* d_edge_v, int edges, PR_RST& mem_mag);

#endif // ROOTED_SPANNING_TREE_PR_CUH