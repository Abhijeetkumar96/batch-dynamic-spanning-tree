#include "dynamic_spanning_tree/profile_emc.cuh"
#include "repl_edges/euler_tour.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// #define DEBUG

__global__ 
void roots_find(const int* parent, int* roots, int* d_num_comp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (parent[idx] == idx) {
            int pos = atomicAdd(d_num_comp, 1); 
            roots[pos] = idx; 
        }
    }
}

__global__
void select_edges_kernel(int* d_parent, int* d_flag, int nodes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nodes) {
        // Mark self-loops
        if (idx != d_parent[idx]) {
            d_flag[idx] = true;
        }
    }
}

__global__
void get_edges(int* d_parent, uint64_t* d_super_tree, int* d_flag, int* d_pos, int nodes, int* num_edges) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx == 0) {
		*num_edges = d_pos[nodes - 1];
	}

    if (idx < nodes) {
    	if(d_flag[idx]) {
    		uint32_t key1 = idx;
    		uint32_t key2 = d_parent[idx];
    		uint64_t combined_key = key1;
    		combined_key = (combined_key << 32) | key2;
    		d_super_tree[d_pos[idx] - 1] = combined_key;
      	}
    }
}

void emc_tour(int* d_parent, int num_nodes) {

	#ifdef DEBUG
		std::cout << "Printing parent array from emc_tour:\n";
		print_device_array(d_parent, num_nodes);
	#endif

	int* d_num_comp;
	int* d_roots;

	CUDA_CHECK(cudaMallocManaged(&d_num_comp, sizeof(int)), 
    	"Failed to allocate d_num_comp");
	CUDA_CHECK(cudaMallocManaged(&d_roots, sizeof(int) * num_nodes), 
    	"Failed to allocate d_roots");

	*d_num_comp = 0;

	int num_threads = 1024;
    int num_blocks = (num_nodes + num_threads - 1) / num_threads;

	roots_find<<<num_blocks, num_threads>>>(d_parent, d_roots, d_num_comp, num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize roots_find kernel");

	int* d_flag;
	CUDA_CHECK(cudaMalloc(&d_flag, num_nodes * sizeof(int)), 
		"Failed to allocate memory for d_flag");

	CUDA_CHECK(cudaMemset(d_flag, 0, num_nodes * sizeof(int)),
		"Failed to memset d_flag");

	int* d_pos;
	CUDA_CHECK(cudaMalloc(&d_pos, num_nodes * sizeof(int)), 
		"Failed to allocate memory for d_pos");

	int* num_edges;
    CUDA_CHECK(cudaMallocManaged(&num_edges, sizeof(int)), 
    	"Failed to allocate num_edges");

    uint64_t* d_super_tree;
    CUDA_CHECK(cudaMalloc(&d_super_tree, num_nodes * sizeof(uint64_t)), 
		"Failed to allocate memory for d_super_tree");

	auto start = std::chrono::high_resolution_clock::now();
	select_edges_kernel<<<num_blocks, num_threads>>>(d_parent, d_flag, num_nodes);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize select_edges_kernel");

	// Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_pos, num_nodes);
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temp storage");
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_pos, num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    get_edges<<<num_blocks, num_threads>>>(d_parent, d_super_tree, d_flag, d_pos, num_nodes, num_edges);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize get_edges");

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << "num_vert: " << num_nodes << ", num_comps: " << *d_num_comp << std::endl;
    std::cout << "emc phase 1: " << duration << " ms.\n";


    #ifdef DEBUG
    	std::cout << "num_edges: " << *num_edges << std::endl;
    	print_device_edge_list(d_super_tree, *num_edges);
    #endif

	Euler_Tour euler(num_nodes, *num_edges, *d_num_comp);

	// Apply eulerianTour algorithm to root an unrooted tree to get replacement edge.
	cuda_euler_tour(
	    d_super_tree,	// edgelist
	    num_nodes,		// num_vertices
	    *num_edges,		// num_edges
	    d_roots,		// roots
	    *d_num_comp,	// count of roots
	    euler);
}