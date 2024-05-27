#ifndef PATH_REVERSAL_H
#define PATH_REVERSAL_H

#include <cuda_runtime.h>

#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "repl_edges/repl_edges.cuh"
#include <thrust/device_vector.h>

void path_reversal_ET(dynamic_tree_manager& tree_ds, int* d_first, int* d_last, REP_EDGES& rep_edge_mag, const int& unique_rep_count);
void path_reversal_PR(dynamic_tree_manager& tree_ds, REP_EDGES& rep_edge_mag, thrust::device_vector<int>& onPath, thrust::device_vector<int>& pr_arr, thrust::device_vector<int>& pr_arr_size, int log_2_size);

#endif // PATH_REVERSAL_H