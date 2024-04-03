#ifndef ROOTED_SPANNING_TREE_PR_CUH
#define ROOTED_SPANNING_TREE_PR_CUH

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

void RootedSpanningTree(uint64_t* d_edgelist, const int numVert, const int numEdges);

#endif // ROOTED_SPANNING_TREE_PR_CUH