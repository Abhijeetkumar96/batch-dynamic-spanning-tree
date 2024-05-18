#ifndef PATH_REVERSAL_H
#define PATH_REVERSAL_H

#include <cuda_runtime.h>

#include "PR-RST/pr_rst_util.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"

void path_reversal(
	dynamic_tree_manager& tree_ds, 
	int* d_first, int* d_last, 
	PR_RST& pr_resource_mag, 
	const int& unique_rep_count);

#endif // PATH_REVERSAL_H