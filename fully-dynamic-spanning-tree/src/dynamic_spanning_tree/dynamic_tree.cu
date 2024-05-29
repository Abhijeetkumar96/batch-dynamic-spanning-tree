/******************************************************************************
* Functionality: Fully Dynamic Tree driver
 ******************************************************************************/

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <set>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Project Specific Libraries
//---------------------------------------------------------------------
#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"

#include "dynamic_spanning_tree/dynamic_tree.cuh"

#include "eulerian_tour/connected/euler_tour.cuh"
#include "eulerian_tour/disconnected/euler_tour.cuh"

#include "PR-RST/shortcutting.cuh"

#include "path_reversal/path_reversal.cuh"

#include "repl_edges/repl_edges.cuh"
#include "repl_edges/super_graph.cuh"
#include "repl_edges/hooking_shortcutting.cuh"

// #define DEBUG
// #define CHECKER

//---------------------------------------------------------------------
// CUDA Kernels
//---------------------------------------------------------------------
__global__
void update_rep_map(int* d_unique_rep, int* d_rep_map, int num_comp) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_comp) {
        // printf("tid: %d, unique_rep: %d, rep_map pos: %d\n", tid, d_unique_rep[tid], tid);
    	d_rep_map[d_unique_rep[tid]] = tid;
    }
}

__global__ 
void find_roots(const int* parent, int* roots, int* d_num_comp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (parent[idx] == idx) {
            int pos = atomicAdd(d_num_comp, 1); 
            roots[pos] = idx; 
        }
    }
}

void repair_spanning_tree(dynamic_tree_manager& tree_ds, bool is_deletion) {

	int* d_rep 			= 	tree_ds.d_parent;
	int* d_unique_rep 	= 	tree_ds.d_unique_rep;
	int* d_rep_map 		= 	tree_ds.d_rep_map;
	int num_vert 		= 	tree_ds.num_vert;

	#ifdef DEBUG
		std::cout << "parent array after deleting edges:\n";
		print_device_array(d_rep, num_vert);
	#endif

	// if is insertion operation, then all parent arrays are same
	if(!is_deletion) {
		CUDA_CHECK(
			cudaMemcpy(
				tree_ds.d_org_parent, 
				tree_ds.new_parent, 
				num_vert * sizeof(int),  
				cudaMemcpyDeviceToDevice), 
			"Failed to copy d_parent to device"
		);

		CUDA_CHECK(
			cudaMemcpy(
				tree_ds.d_parent, 
				tree_ds.new_parent, 
				num_vert * sizeof(int),  
				cudaMemcpyDeviceToDevice), 
			"Failed to copy d_parent to device"
		);
	}

	CUDA_CHECK(cudaMemset(d_unique_rep, 0, sizeof(int) * num_vert), 
		"Failed to memset d_unique_rep");

	CUDA_CHECK(cudaMemset(d_rep_map, 0, sizeof(int) * num_vert), 
		"Failed to memset d_rep_map");
		
	// We need to copy the current parent array (tree_ds.d_parent (updated parent array after deleting edges)) 
	// to the new parent array (tree_ds.new_parent).
	// This is necessary because tree_ds.d_parent will be used for pointer jumping operations,
	// during which it will be transformed into the representative array.
	// tree_ds.new_parent array will be used during Path Reversal Process
	CUDA_CHECK(cudaMemcpy(tree_ds.new_parent, tree_ds.d_parent, num_vert * sizeof(int), cudaMemcpyDeviceToDevice), 
        "Failed to copy d_parent to device");

    // if is_deletion operation, then check if the input tree is connected or not
    // and find the root value if connected
    bool is_connected = false;
    int root = -1;

    // checking if the original parent array, before deleting the edges, was connected or not
	is_connected = is_tree_or_forest(tree_ds.d_org_parent, num_vert, root);

	if(is_connected and !is_deletion) {
		std::cerr << "The input tree is already connected; no repair needed.\n";
		return;
	}
	
	// 1. find all unique representatives 
	int* d_roots    = tree_ds.d_unique_rep;
	int* d_num_comp = nullptr;

    CUDA_CHECK(cudaMallocManaged((void**)&d_num_comp, sizeof(int)), 
    	"Failed to allocate d_num_comp");

    *d_num_comp = 0;

    int block_size = 1024;
    int num_blocks = (num_vert + block_size - 1) / block_size;

    // 2. calculate num_components and identify the roots
    auto start = std::chrono::high_resolution_clock::now();
    // d_parent is the updated parent array
    find_roots<<<num_blocks, block_size>>>(tree_ds.d_parent, d_roots, d_num_comp, num_vert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize find_roots kernel");
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Finding unique comps", duration);

	int num_comp = *d_num_comp;

	if(is_connected)
		std::cout << "Root: " << root << std::endl;
	else
		std::cout << "Input tree not connected. num_comp: " << num_comp << "\n";

	#ifdef DEBUG
		std::cout << "num_comp: " << num_comp << std::endl;
		std::cout << "d_unique_rep array:\n";
		print_device_array(d_unique_rep, num_comp);

		std::cout << "d_roots array:\n";
		print_device_array(d_roots, num_comp);
	#endif
	
	// 3. Do pointer jumping over parent array to update representative array.
	start = std::chrono::high_resolution_clock::now();
	pointer_jumping(d_rep, num_vert);

	stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Pointer Jumping", duration);

    #ifdef DEBUG
		std::cout << "After doing pointer_jumping:\n";
		print_device_array(d_rep, num_vert);
	#endif

	num_blocks = (num_comp + block_size - 1) / block_size;
	
	// 4. update representative mapping array
	start = std::chrono::high_resolution_clock::now();
	update_rep_map<<<num_blocks, block_size>>>(d_unique_rep, d_rep_map, num_comp);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_rep_map");

	stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Update rep array", duration);
	
	#ifdef DEBUG	
		std::cout << "d_rep_map array:\n";
		print_device_array(d_rep_map, num_vert);
	#endif
    
	REP_EDGES rep_edge_mag(num_comp);

	// get_replacement_edges(tree_ds, rep_edge_mag, num_comp, is_deletion);
	
	// 5. Get the replacement edges
	std::cout << "Selected Algorithm: " << rep_edge_algo << "_" << path_rev_algo << std::endl;

	if(rep_edge_algo == "SG_PR") {
		std::cout << "Finding Replacement edges using SG.\n";
        super_graph(tree_ds, rep_edge_mag, is_deletion);
    } else if(rep_edge_algo == "HS_ET") {
    	std::cout << "Finding Replacement edges using HS.\n";
        hooking_shortcutting(tree_ds, rep_edge_mag, is_deletion);
    } else {
        std::cerr << "Unrecognized algorithm to use\n";
        return;
    }

	int* unique_super_graph_edges = tree_ds.super_graph_edges;

	if(*unique_super_graph_edges < 1) {
        std::cerr << "No cross edges found to connect the graphs.\n";
        return;
    }

    // 6. reverse the path 
    if (path_rev_algo == "ET") {
    	std::cout << "Reversing the paths using eulerian tour " << std::endl;
	    int *d_first = nullptr;
    	int *d_last = nullptr;

    	// Declare the pointer outside of the if-else block
    	mce::EulerianTour* mce_euler_mag = nullptr;
    	sce::EulerianTour* sce_euler_mag = nullptr;
    
	    // 5. Find the eulerian Tour 
	    if(!is_connected) {
	    	// std::cout << "Started Multi-component eulerian Tour.\n";
	    	// std::cout << "num_comp: " << num_comp << "\n";
	        mce_euler_mag = new mce::EulerianTour(num_vert, num_comp);
	        // if the tree is not connected, then it is forest
	        // find the Euler tour for the forest (mce is multi-component eulerian tour)
	        auto start = std::chrono::high_resolution_clock::now();
	        mce::cal_first_last(tree_ds.d_org_parent, d_roots, d_rep, d_rep_map, num_vert, num_comp, mce_euler_mag); 
	        d_first = mce_euler_mag->d_new_first;
	        d_last  = mce_euler_mag->d_new_last;

	        auto stop = std::chrono::high_resolution_clock::now();
	        auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

	        add_function_time("ET: First & Last", duration);
	    } else {
	        sce_euler_mag = new sce::EulerianTour(num_vert);
	        // Find the Euler tour for the tree (sce is single-component eulerian tour)
	        auto start = std::chrono::high_resolution_clock::now();
	        sce::cal_first_last(root, tree_ds.d_org_parent, sce_euler_mag);
	        d_first = sce_euler_mag->new_first;
	        d_last  = sce_euler_mag->new_last;

	        auto stop = std::chrono::high_resolution_clock::now();
	        auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

	        add_function_time("ET: First & Last", duration);
	    }

	    start = std::chrono::high_resolution_clock::now();
		path_reversal_ET(tree_ds, d_first, d_last, rep_edge_mag, num_comp);
		stop = std::chrono::high_resolution_clock::now();
	    duration = std::chrono::duration<double, std::milli>(stop - start).count();

	    add_function_time("Path Reversal", duration);

	    // Free the memory allocated for euler_mag 
	    if(mce_euler_mag)
	    	delete mce_euler_mag;
	    if(sce_euler_mag)
	    	delete sce_euler_mag;

	} else if(path_rev_algo == "PR") {
		std::cout << "Reversing the paths using PR module " << std::endl;
		int log_2_size    =  std::ceil(std::log2(num_vert));
	    long long pr_size =  std::ceil(num_vert * 1LL * log_2_size);
	    
	    thrust::device_vector <int> parent_pr(num_vert), parent_pr_tmp(num_vert);
	    thrust::copy(tree_ds.new_parent,tree_ds.new_parent + num_vert, parent_pr.begin());

		thrust::device_vector <int> pr_arr(pr_size);
	    thrust::device_vector <int> tobe_rep(num_vert), pr_arr_size(num_vert);

	    auto start = std::chrono::high_resolution_clock::now();
	    
	    Shortcut(num_vert, tree_ds.num_edges, log_2_size, 
	    thrust::raw_pointer_cast(parent_pr.data()),
	    thrust::raw_pointer_cast(parent_pr_tmp.data()),
	    thrust::raw_pointer_cast(pr_arr.data()),
	    thrust::raw_pointer_cast(tobe_rep.data()),
	    thrust::raw_pointer_cast(pr_arr_size.data())
		);

		auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

	    add_function_time("PR_Shortcut", duration);

	    thrust::device_vector <int> onPath(num_vert);
		path_reversal_PR(tree_ds, rep_edge_mag, onPath, pr_arr, pr_arr_size, log_2_size);
	} else {
		std::cerr << "Invalid selection.\n";
		return;
	}

    CUDA_CHECK(cudaFree(d_num_comp), "Failed to free d_num_comp");
}

// ====[ End of driver Code ]====