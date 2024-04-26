#ifndef REVERSE_PATH_PR_H
#define REVERSE_PATH_PR_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

void ReversePaths(
	int vertices,
	int edges,
	int log_2_size,
	int *d_OnPath,
	int *d_new_OnPath,
	int *d_pr_arr,
	int *d_parent_ptr,
	int *d_new_parent_ptr,
	int *d_index_ptr,
	int *d_pr_size_ptr);

#endif // REVERSE_PATH_PR_H