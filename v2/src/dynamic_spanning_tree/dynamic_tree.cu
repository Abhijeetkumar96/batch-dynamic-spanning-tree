#include <cuda_runtime.h>

#include "super_graph/super_graph.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "common/cuda_utility.cuh"
#include "common/Timer.hpp"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"
#include "dynamic_spanning_tree/path_reversal.cuh"

__global__
void update_rep_map(int* d_unique_rep, int* d_rep_map, int unique_rep_count) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < unique_rep_count) {
        // printf("tid: %d, unique_rep: %d, rep_map pos: %d\n", tid, d_unique_rep[tid], tid);
    	d_rep_map[d_unique_rep[tid]] = tid;
    }
}

void repair_spanning_tree(const std::vector<int>& roots, dynamic_tree_manager& tree_ds, EulerianTour& euler_tour) {

	int* d_rep = tree_ds.d_parent;
	int* d_rep_dup = tree_ds.d_rep_dup;
	int* d_unique_rep = tree_ds.d_unique_rep;
	int* d_rep_map = tree_ds.d_rep_map;
	int num_vert = tree_ds.num_vert;
	
	Timer myTimer;
    myTimer.start();
    std::cout << "Timer started" << std::endl;

	#ifdef DEBUG
		std::cout << "parent array after deleting edges:\n";
		print_device_array(d_rep, num_vert);
	#endif

	// 1. find eulerian tour
	cal_first_last(roots[0], tree_ds.d_org_parent, euler_tour);

	// 2. Do pointer jumping over parent array to update representative array.
	pointer_jumping(d_rep, tree_ds.num_vert);
	
	CUDA_CHECK(cudaMemcpy(d_rep_dup, d_rep, sizeof(int) * tree_ds.num_vert, cudaMemcpyDeviceToDevice),
		"Failed to copy d_rep array to new d_rep_array");

	#ifdef DEBUG
		std::cout << "After doing pointer_jumping:\n";
		print_device_array(d_rep, num_vert);
	#endif
	
	// 3. find unique in the d_rep array
	// send a copy of d_rep.
	int unique_rep_count = 0;
	find_unique(d_rep_dup, d_unique_rep, tree_ds.num_vert, unique_rep_count);
	std::cout << "unique_rep_count: " << unique_rep_count << std::endl;
	
	#ifdef DEBUG
		std::cout << "d_rep array after find_unique:\n";
		print_device_array(d_rep, tree_ds.num_vert);
		
		std::cout << "d_unique_rep array:\n";
		print_device_array(d_unique_rep, unique_rep_count);

	#endif

	int numThreads = 1024;
	int numBlocks = (unique_rep_count + numThreads - 1) / numThreads;
	
	// update rep_map
	update_rep_map<<<numBlocks, numThreads>>>(d_unique_rep, d_rep_map, unique_rep_count);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_rep_map");
	
	#ifdef DEBUG	
		std::cout << "d_rep_map array:\n";
		print_device_array(d_rep_map, num_vert);
	#endif
    
    myTimer.pause();
	PR_RST resource_mag(unique_rep_count);
	myTimer.resume();
	// weed out self loops and duplicates and get the replacement edges
	get_replacement_edges(tree_ds, resource_mag, unique_rep_count);

	path_reversal(tree_ds, euler_tour, resource_mag, unique_rep_count);

    std::cout << "Total elapsed time for dynamic_spanning_tree repair: " << myTimer.getElapsedMilliseconds() << " ms" << std::endl;

}