#ifndef PATH_REVERSAL_H
#define PATH_REVERSAL_H

#include <cuda_runtime.h>
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"

void path_reversal(dynamic_tree_manager& tree_ds, EulerianTour& euler_tour, PR_RST& pr_resource_mag, const int& unique_rep_count);

#endif // PATH_REVERSAL_H