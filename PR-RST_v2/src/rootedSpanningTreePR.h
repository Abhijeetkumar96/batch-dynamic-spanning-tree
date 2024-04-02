#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

std::vector<int> RootedSpanningTree(uint64_t* d_edgelist, const int numVert, const int numEdges);

bool findEdge(const std::vector<std::pair<int, int>> &edge_stream, std::pair<int, int> target);
bool validateRST(const std::vector<int> &parent);
int treeDepth(const std::vector<int> &parent);

