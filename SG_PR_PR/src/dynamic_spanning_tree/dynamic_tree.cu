#include <set>
#include <cuda_runtime.h>

#include "super_graph/super_graph.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "common/cuda_utility.cuh"
#include "common/Timer.hpp"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "PR-RST/pr_rst_util.cuh"
#include "dynamic_spanning_tree/path_reversal.cuh"
#include "PR-RST/shortcutting.cuh"

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

// #define DEBUG
// #define CHECKER

__global__
void update_rep_map(int* d_unique_rep, int* d_rep_map, int unique_rep_count) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < unique_rep_count) {
        // printf("tid: %d, unique_rep: %d, rep_map pos: %d\n", tid, d_unique_rep[tid], tid);
    	d_rep_map[d_unique_rep[tid]] = tid;
    }
}


void repair_spanning_tree(const std::vector<int>& roots, dynamic_tree_manager& tree_ds, EulerianTour& euler_tour) {

	int* d_rep 			= 	tree_ds.d_parent;
	int* d_unique_rep 	= 	tree_ds.d_unique_rep;
	int* d_rep_map 		= 	tree_ds.d_rep_map;
	int num_vert 		= 	tree_ds.num_vert;
	int num_edges 		= 	tree_ds.num_edges;
	
	// Timer myTimer;
    // myTimer.start();
    // std::cout << "Timer started" << std::endl;

	#ifdef DEBUG
		std::cout << "parent array after deleting edges:\n";
		print_device_array(d_rep, num_vert);
	#endif

	auto start = std::chrono::high_resolution_clock::now();
	// 1. find eulerian tour
	cal_first_last(roots[0], tree_ds.d_org_parent, euler_tour);

	auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Eulerian Tour", duration);

	CUDA_CHECK(cudaMemcpy(tree_ds.new_parent, tree_ds.d_parent, num_vert * sizeof(int), cudaMemcpyDeviceToDevice), 
        "Failed to copy d_parent to device");

	start = std::chrono::high_resolution_clock::now();
	// 2. Do pointer jumping over parent array to update representative array.
	pointer_jumping(d_rep, num_vert);

	stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Pointer Jumping", duration);
	
	#ifdef CHECKER
		std::vector<int> h_rep(tree_ds.num_vert);
		CUDA_CHECK(cudaMemcpy(h_rep.data(), d_rep, sizeof(int) * tree_ds.num_vert, cudaMemcpyDeviceToHost),
		"Failed to copy d_rep array to host");

		bool result = compare_arrays(h_rep.data(), tree_ds.parent_array, tree_ds.num_vert);

    	std::cout << "Comparison of cpu p_jump and gpu p_jump: " << (result ? "Equal" : "Not Equal") << std::endl;

	    std::set<int> unique_elements(h_rep.begin(), h_rep.end());
	    std::cout << "Unique representatives after deleting edges: " << unique_elements.size() << "\n";
    	
    	// std::cout << "Unique representatives in sorted order: \n";
    	// for (int element : unique_elements) {
        // 	std::cout << element << " ";
    	// }
    	// std::cout << std::endl;
    #endif

	#ifdef DEBUG
		std::cout << "After doing pointer_jumping:\n";
		print_device_array(d_rep, num_vert);
	#endif
	
	// 3. find unique in the d_rep array
	int unique_rep_count = tree_ds.unique_rep_count;

	#ifdef DEBUG
		std::cout << "unique_rep_count: " << unique_rep_count << std::endl;
		std::cout << "d_unique_rep array:\n";
		print_device_array(d_unique_rep, unique_rep_count);
	#endif

	int numThreads = 1024;
	int numBlocks = (unique_rep_count + numThreads - 1) / numThreads;
	
	start = std::chrono::high_resolution_clock::now();
	// update rep_map
	update_rep_map<<<numBlocks, numThreads>>>(d_unique_rep, d_rep_map, unique_rep_count);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_rep_map");

	stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Update rep array", duration);
	
	#ifdef DEBUG	
		std::cout << "d_rep_map array:\n";
		print_device_array(d_rep_map, num_vert);
	#endif
    
    // myTimer.pause();
	PR_RST resource_mag(unique_rep_count);
	// myTimer.resume();
	// weed out self loops and duplicates and get the replacement edges

	get_replacement_edges(tree_ds, resource_mag, unique_rep_count);

	int* unique_super_graph_edges = tree_ds.super_graph_edges;

	if(*unique_super_graph_edges < 1) {
        // std::cerr << "No cross edges found to connect the graphs.\n";
        return;
    }

    // 
    int log_2_size = std::ceil(std::log2(num_vert));
    long long pr_size = std::ceil(num_vert * 1LL * log_2_size);
    thrust::device_vector <int> parent_pr(num_vert), parent_pr_tmp(num_vert);
    thrust::copy(tree_ds.new_parent,tree_ds.new_parent + num_vert, parent_pr.begin());

	thrust::device_vector <int> pr_arr(pr_size);
    thrust::device_vector <int> tobe_rep(num_vert), pr_arr_size(num_vert);

    // std::cout << "parent array:\n";
    // for(auto i : parent_pr) {
    // 	std::cout << i << " ";
    // }
    // std::cout << std::endl;

    Shortcut(num_vert, num_edges, log_2_size, 
    thrust::raw_pointer_cast(parent_pr.data()),
    thrust::raw_pointer_cast(parent_pr_tmp.data()),
    thrust::raw_pointer_cast(pr_arr.data()),
    thrust::raw_pointer_cast(tobe_rep.data()),
    thrust::raw_pointer_cast(pr_arr_size.data())
	);

	// std::cout<<"PR arr : \n";
	// for(int i=0;i<num_vert;i++)
	// {
	// 	std::cout<<i<<" : ";
	// 	for(int j=0;j<log_2_size;j++)
	// 	{
	// 		std::cout<<pr_arr[i*log_2_size+j]<<" ";
	// 	}
	// 	std::cout<< " | " <<pr_arr_size[i]<<"\n";
	// }

    start = std::chrono::high_resolution_clock::now();

    thrust::device_vector <int> onPath(num_vert);

	path_reversal(tree_ds, euler_tour, resource_mag, unique_rep_count, onPath, pr_arr, pr_arr_size, log_2_size);
	stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Path Reversal", duration);

    // std::cout << "Total elapsed time for dynamic_spanning_tree repair: " << myTimer.getElapsedMilliseconds() << " ms" << std::endl;

    // validation
    // do pointer_jumping and is_unique
}
