#include <cuda_runtime.h>

#include "dynamic_spanning_tree/path_reversal.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "common/cuda_utility.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"

__global__
void update_parent_kernel(
	int* d_new_parent, 
	int* d_parent, 
	int* d_first, 
	int* d_last, 
	int* d_interval, 
	int* d_unique_rep, 
	int* d_rep_array, 
	int* d_repMap, 
	int* d_edge_u, 
	int* d_parent_u, 
	int n, int m) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n) {

		int k = d_repMap[d_rep_array[tid]];
		int starting_node = d_unique_rep[k];
		int ending_node = d_interval[k];
		
		if((d_first[starting_node] < d_first[tid]) && 
			(d_first[tid] <= d_first[ending_node]) && 
			(d_last[starting_node] >= d_last[tid]) && 
			d_last[ending_node] <= d_last[tid]) {

			int p = d_parent[tid];
			d_new_parent[p] = tid;
		}

	}

	if(tid > 0 and tid < m) {

		d_new_parent[d_edge_u[tid]] = d_parent_u[tid];
	}
}

__global__
void generate_interval_kernel(
	int* d_edge_u, 
	int* d_interval, 
	int* d_repMap, 
	int*d_rep_array, 
	int root, 
	int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //n <-- SuperGraph size
    if(tid < n)
    {
    	int edge = d_edge_u[tid];
    	d_interval[d_repMap[d_rep_array[edge]]] = edge;
    }
    if(tid == 0)
    	d_interval[tid] = root;
}

void path_reversal(
	dynamic_tree_manager& tree_ds, 
	EulerianTour& euler_tour, 
	PR_RST& pr_resource_mag, 
	const int& unique_rep_count) {

    int* edge_u = pr_resource_mag.d_edge_u;
    int* parent_u = pr_resource_mag.d_parent_u;
	
	int* interval = pr_resource_mag.interval;
	int* d_rep_map = tree_ds.d_rep_map;
	int* d_rep = tree_ds.d_parent;
	int root = 0;
	int n = unique_rep_count;
	//n == uniqueRep array size

	int numThreads = 1024;
	int numBlocks = (n + numThreads - 1) / numThreads;    

	generate_interval_kernel<<<numBlocks, numThreads>>>(
				edge_u,
        		interval,
        		d_rep_map,
        		d_rep,
				root,
				n);
	
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after generate_interval_kernel");
	// update_parent();
	int p_size = tree_ds.num_vert;
	int* h_parent = tree_ds.d_parent;
	int* new_parent = tree_ds.new_parent;
	int* first = euler_tour.new_first;
	int* last = euler_tour.new_last;
	int* d_unique_rep = tree_ds.d_unique_rep;

	numBlocks = (p_size + numThreads - 1) / numThreads;    

	update_parent_kernel<<<numBlocks, numThreads>>>(
		new_parent,
		h_parent,
		first,
		last,
		interval,
		d_unique_rep,
		d_rep,
		d_rep_map,
		edge_u,
		parent_u,
		p_size,
		unique_rep_count);

	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_parent_kernel");
	#ifdef DEBUG
		std::cout << "New parent array:\n";
		print_device_array(new_parent, p_size);
	#endif
}

