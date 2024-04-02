#ifndef RE_ROOT_H
#define RE_ROOT_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

void ReRoot(
	int vertices,
	int edges,
	int log_2_size,
	int iter_number,
	int *d_OnPath,
	int *d_new_OnPath,
	int *d_pr_arr,
	int *d_parent_ptr,
	int *d_new_parent_ptr,
	int *d_index_ptr,
	int *d_pr_size_ptr,
	int *d_marked_parent,
	int *d_ptr);

#endif // RE_ROOT_H