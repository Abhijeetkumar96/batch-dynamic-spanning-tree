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

void Shortcut(
	int vertices,
	int edges,
	int log_2_size,
	int *d_next,
	int *d_new_next,
	int *d_pr_arr,
	int *d_ptr,
	int *d_pr_size_ptr
);