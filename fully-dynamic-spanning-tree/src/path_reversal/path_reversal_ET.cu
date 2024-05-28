#include <cuda_runtime.h>

#include "common/cuda_utility.cuh"

#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "path_reversal/path_reversal.cuh"

#define DEBUG

__global__
void generate_interval_kernel(
	int* d_edge_u, 
	int* d_interval, 
	int* d_repMap, 
	int*d_rep_array, 
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
        	(out_time[tid] <= out_time[x])) {

			int p = d_parent[tid];
			printf("for tid: %d, parent: %d, \n", tid, p);
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
		printf("For tid:%d, edge_u: %d, parent_u:%d\n", tid, edge_u[tid], parent_u[tid]);
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

void path_reversal_ET(
	dynamic_tree_manager& tree_ds, 
	int* d_first, int* d_last,
	REP_EDGES& rep_edge_mag, 
	const int& unique_rep_count) {

    int* edge_u 	= rep_edge_mag.d_edge_u;
    int* parent_u 	= rep_edge_mag.d_parent_u;
	int* d_interval = rep_edge_mag.interval;
	int* d_rep_map 	= tree_ds.d_rep_map;
	int* d_rep 		= tree_ds.d_parent;
	
	int n = unique_rep_count;
	
	//n == uniqueRep array size

	// std::cout << "Executing path_reversal:\n";

	g_verbose = true;

	int numThreads = 1024;
	int numBlocks = (n + numThreads - 1) / numThreads;    

	if(g_verbose) {
		std::cout << "Printing from path_reversal function:\n";
        int* h_first = new int[tree_ds.num_vert];
        int* h_last =  new int[tree_ds.num_vert];
        
        // Step 2: Copy data from device to host
        cudaMemcpy(h_first, d_first, tree_ds.num_vert * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_last, d_last, tree_ds.num_vert * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Node\tFirst\tLast\n";
        
        for (int i = 0; i < tree_ds.num_vert; ++i) {
            std::cout << "Node " << i << ": " << h_first[i] << "\t" << h_last[i] << "\n";
        }    
    }

	generate_interval_kernel<<<numBlocks, numThreads>>>(
				edge_u,
        		d_interval,
        		d_rep_map,
        		d_rep,
				n);
	
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after generate_interval_kernel");
	
	if(g_verbose) {
		std::cout << "Intervals: " << std::endl;
		print_interval<<<1,1>>>(d_interval, n);
		cudaDeviceSynchronize();
		std::cout << std::endl;
	}

	int p_size        = tree_ds.num_vert;     // parent size or numVert in the original graph
	int* d_parent     = tree_ds.d_org_parent; // original parent array before deleting any edges
	int* new_parent   = tree_ds.new_parent;   // parent array after deleting edges
	int* d_unique_rep = tree_ds.d_unique_rep; // all the unique representatives in the forest


	numBlocks = (p_size + numThreads - 1) / numThreads;    

	update_parent_kernel<<<numBlocks, numThreads>>>(
		new_parent,			// 1
		d_parent,			// 2
		d_first,			// 3
		d_last,				// 4
		d_interval,			// 5
		d_unique_rep,		// 6
		d_rep,				// 7
		d_rep_map,			// 8
		edge_u,				// 9
		parent_u,			// 10
		p_size,				// 11
		unique_rep_count);	// 12

	#ifdef DEBUG
		CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_parent");
		std::cout << "Parent array before final reversal:" << std::endl;
		print_parent<<<1,1>>>(new_parent, p_size);
		CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_parent");
		std::cout << std::endl;
	#endif

	// sg_size is super_graph parent array size / numVert in superGraph
    int sg_size = rep_edge_mag.num_vert;
    numBlocks = (sg_size + numThreads - 1) / numThreads;

	reverse_new_parents<<<numBlocks, numThreads>>>(
		edge_u, 
        parent_u,
        new_parent, 
        sg_size);

	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_parent_kernel");
	
	if(g_verbose) {
		std::cout << "New parent array:" << std::endl;
		print_parent<<<1,1>>>(new_parent, p_size);
		CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_parent");
		std::cout << std::endl;
	}
}

