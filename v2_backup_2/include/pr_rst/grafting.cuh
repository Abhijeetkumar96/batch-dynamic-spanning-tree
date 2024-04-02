#ifndef GRAFTING_H
#define GRAFTING_H

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

void Graft(
	int vertices,
	int edges,
	uint64_t* d_edge_list,
	int *d_ptr,
	int *d_winner_ptr,
	int *d_marked_parent,
	int *d_OnPath,
	int *d_flag);

#endif // GRAFTING_H