#ifndef PATH_REVERSAL_H
#define PATH_REVERSAL_H

#include <cuda_runtime.h>
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

void path_reversal(dynamic_tree_manager& tree_ds, EulerianTour& euler_tour, PR_RST& pr_resource_mag, const int& unique_rep_count, thrust::device_vector<int> &onPath, thrust::device_vector<int> &pr_arr,thrust::device_vector<int> &pr_arr_size,int log_2_size);

#endif // PATH_REVERSAL_H