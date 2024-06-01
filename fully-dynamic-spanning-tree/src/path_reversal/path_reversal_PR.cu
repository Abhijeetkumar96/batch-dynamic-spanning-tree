#include <cuda_runtime.h>

#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"

#include "dynamic_spanning_tree/dynamic_tree.cuh"

#include "path_reversal/path_reversal.cuh"
#include "PR-RST/reversePaths.cuh"

__global__
void reverse_new_parents_(
	int* edge_u, 
	int* parent_u, 
	int* new_parent, 
	int h_size) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < h_size) {
		// printf("[%d,%d]\n",edge_u[tid],parent_u[tid]);
		new_parent[edge_u[tid]] = parent_u[tid];
	}	
}

void path_reversal_PR(
	dynamic_tree_manager& tree_ds, 
	REP_EDGES& rep_edge_mag, 
	thrust::device_vector<int> &onPath,
	thrust::device_vector<int> &pr_arr,
	thrust::device_vector<int> &pr_arr_size,
	int log_2_size) {

	int num_vert  = tree_ds.num_vert;
	int num_edges = tree_ds.num_edges;
    int* edge_u   = rep_edge_mag.d_edge_u;
    int* parent_u = rep_edge_mag.d_parent_u;
	
	//n == uniqueRep array size

	std::cout << "Executing path_reversal:\n";

	thrust::device_vector <int> onPathCpy(num_vert);
	thrust::device_vector <int> parent_pr_tmp(num_vert);
	thrust::device_vector <int> us1(num_vert);

	int numThreads = 1024;

	auto start = std::chrono::high_resolution_clock::now();

	int p_size = tree_ds.num_vert;
	int* new_parent = tree_ds.new_parent;

    ReversePaths(num_vert, num_edges, log_2_size, 
        thrust::raw_pointer_cast(onPath.data()),
        thrust::raw_pointer_cast(onPathCpy.data()),
        thrust::raw_pointer_cast(pr_arr.data()),
        new_parent,          // changes reflected
        thrust::raw_pointer_cast(parent_pr_tmp.data()),
        thrust::raw_pointer_cast(us1.data()),
        thrust::raw_pointer_cast(pr_arr_size.data())   
    );

    // if(g_verbose) {
    	// std::cout<<"Parent after: ";
		// for(auto i : n_parent)
		// {
		// 	std::cout<<i<<" ";
		// }
		// std::cout<<"\n";
    // }

	// h_size is super_graph parent array size
    int h_size = rep_edge_mag.num_vert;
    int numBlocks = (h_size + numThreads - 1) / numThreads;

	reverse_new_parents_<<<numBlocks, numThreads>>>(
		edge_u, 
        parent_u,
        new_parent, 
        h_size);

	CUDA_CHECK(cudaDeviceSynchronize(), 
		"Failed to synchronize after update_parent_kernel");
	
	auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Path Reversal", duration);

	// g_verbose = false;

	if(g_verbose) {
		std::cout << "New parent array:\n";
		std::vector<int> h_new_parent(p_size);
		CUDA_CHECK(cudaMemcpy(h_new_parent.data(), new_parent, p_size * sizeof(int), cudaMemcpyDeviceToHost), 
			"Failed to copy back new parent array");
		int j = 0;
		for(auto i : h_new_parent) {
			std::cout << "new_parent[" << j++ << "]= " << i << "\n";
		}
	}
}

