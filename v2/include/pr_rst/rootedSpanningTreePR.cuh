#ifndef ROOTED_SPANNING_TREE_PR_H
#define ROOTED_SPANNING_TREE_PR_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

void RootedSpanningTree(
    RST_Resource_manager& pr_resources, 
    int* d_u_ptr, 
    int* d_v_ptr, 
    const int numVert, 
    const int numEdges);

#endif // ROOTED_SPANNING_TREE_PR_H
