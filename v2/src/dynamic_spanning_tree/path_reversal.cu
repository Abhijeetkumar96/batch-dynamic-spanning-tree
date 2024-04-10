#include <cuda_runtime.h>

#include "dynamic_spanning_tree/path_reversal.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "common/cuda_utility.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"

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
    if(tid < n) {
    	int u = d_edge_u[tid];
		int rep_u = d_rep_array[u];
		int mapped_rep = d_repMap[rep_u];
		// printf("u: %d, rep_u: %d, mapped_rep: %d\n", u, rep_u, mapped_rep);
        d_interval[mapped_rep] = u;
    }
}

__global__
void print_interval(int* d_interval, int count) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid == 0) {
		for(int i = 0; i < count; ++i) {
			printf("interval[%d]: %d\n", i, d_interval[i]);
		}
	}
}

__global__
void update_parent_kernel(
	int* d_new_parent, 
	int* d_parent, 
	int* in_time, 
	int* out_time, 
	int* d_interval, 
	int* d_unique_rep, 
	int* d_rep_array, 
	int* d_repMap, 
	int* d_edge_u, 
	int* d_parent_u, 
	int n, int m) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n) {

		int x = d_rep_array[tid];
        int y = d_interval[d_repMap[x]];

        // printf("for vert: %d, starting_node: %d, ending_node: %d\n", tid, x, y);
		if(
        	(in_time[tid] > in_time[x]) && 
        	(in_time[tid] <= in_time[y]) && 
        	(out_time[tid] >= out_time[y]) && 
        	(out_time[tid] <= out_time[x]) ){

			int p = d_parent[tid];
			// printf("for tid: %d, parent: %d, \n", tid, p);

			d_new_parent[p] = tid;
		}

	}
}

__global__
void reverse_new_parents(
	int* edge_u, 
	int* parent_u, 
	int* new_parent, 
	int h_size) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < h_size) {
		new_parent[edge_u[tid]] = parent_u[tid];
	}	
}

__global__
void print_parent(int* new_parent, int p_size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid == 0) {
		for(int i = 0; i < p_size; ++i) {
			printf("new_parent[%d]: %d\n", i, new_parent[i]);
		}
	}
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
	
	if(g_verbose) {
		print_interval<<<1,1>>>(interval, n);
		cudaDeviceSynchronize();
	}

	int p_size = tree_ds.num_vert;
	int* d_parent = tree_ds.d_org_parent;
	int* new_parent = tree_ds.new_parent;
	int* first = euler_tour.new_first;
	int* last = euler_tour.new_last;
	int* d_unique_rep = tree_ds.d_unique_rep;

	numBlocks = (p_size + numThreads - 1) / numThreads;    

	update_parent_kernel<<<numBlocks, numThreads>>>(
		new_parent,
		d_parent,
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

	// h_size is super_graph parent array size
    int h_size = pr_resource_mag.num_vert;
    numBlocks = (h_size + numThreads - 1) / numThreads;

	reverse_new_parents<<<numBlocks, numThreads>>>(
		edge_u, 
        parent_u,
        new_parent, 
        h_size);

	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_parent_kernel");
	
	g_verbose = false;

	if(g_verbose) {
		std::cout << "New parent array:\n";
		print_parent<<<1,1>>>(new_parent, p_size);
		cudaDeviceSynchronize();
	}
}

