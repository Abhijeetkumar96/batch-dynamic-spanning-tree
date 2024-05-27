#ifndef REP_EDGES_MANAGER_CUH
#define REP_EDGES_MANAGER_CUH

#include <cuda_runtime.h>
#include "common/cuda_utility.cuh"

class REP_EDGES {

public:
	int* interval;
	
	// original replacement edges
	int* d_edge_u;
	int* d_parent_u;

	int num_vert;

	REP_EDGES(int num_vertices) : num_vert(num_vertices) {

		size_t size = num_vert * sizeof(int); // For n vertices

		CUDA_CHECK(cudaMalloc((void**)&interval, size),	  "Failed to allocate memory for interval");
		CUDA_CHECK(cudaMalloc((void**)&d_edge_u, size),	  "Failed to allocate memory for d_edge_u");
		CUDA_CHECK(cudaMalloc((void**)&d_parent_u, size), "Failed to allocate memory for d_parent_u");
	}
	~REP_EDGES() {
		// Free allocated device memory
	    CUDA_CHECK(cudaFree(interval),   "Failed to free interval");
	    CUDA_CHECK(cudaFree(d_edge_u),   "Failed to free d_edge_u");
	    CUDA_CHECK(cudaFree(d_parent_u), "Failed to free d_parent_u");
	}
};

#endif // REP_EDGES_MANAGER_CUH