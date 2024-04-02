#ifndef ROOTED_SPANNING_TREE_PR_H
#define ROOTED_SPANNING_TREE_PR_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

#include "pr_rst.cuh"

void RootedSpanningTree(
    RST_Resource_manager& pr_resources, 
    uint64_t* d_edge_list,
    const int numVert, 
    const int numEdges);

#endif // ROOTED_SPANNING_TREE_PR_H
