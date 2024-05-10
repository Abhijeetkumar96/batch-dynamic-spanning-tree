#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "PR-RST/rootedSpanningTreePR.cuh"
#include "PR-RST/grafting.cuh"
#include "PR-RST/reRoot.cuh"
#include "PR-RST/pr_rst_util.cuh"
#include "PR-RST/shortcutting.cuh"

#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"

// #define DEBUG

__global__ 
void init(int *arr, int *rep, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
	{
		arr[tid] = tid;
		rep[tid] = tid;
	}
}

int* RootedSpanningTree(int* d_u_arr, int* d_v_arr, const int numVert, const int numEdges) {

	int n = numVert;
	int vertices = n;
	int edges = numEdges;

	// std::cout << "No. of vertices = " << vertices << std::endl;
	// std::cout << "No. of edges = " << edges << std::endl;

	thrust::device_vector<int> d_winner(n, 0);

	int *d_winner_ptr = thrust::raw_pointer_cast(d_winner.data());

	// Update values for pointerJumping
	// std::cout << "log2(n) = " << std::log2(n) << std::endl;
	int log_2_size = std::ceil(std::log2(n));
	long long pr_size = std::ceil(n * 1LL * log_2_size) * sizeof(int);
	// std::cout << "pr_size = " << pr_size << std::endl;
	
	long long size = n * 1LL * sizeof(int); // For n vertices

	// std::cout << "size: " <<  size << std::endl;

	// std::cout <<"Received edgelist:\n";

	// print_device_array(d_u_ptr, numEdges);
	// print_device_array(d_v_ptr, numEdges);

	int *d_ptr;
	int *d_parent_ptr;
	int *d_new_parent_ptr;
	int *d_pr_arr;
	int *d_label;
	int *d_OnPath;
	int *d_new_OnPath;
	int *d_rep;
	int *d_marked_parent;
	int *d_next;
	int *d_new_next;
	int *d_index_ptr;
	int *d_pr_size_ptr;

	CUDA_CHECK(cudaMalloc((void **)&d_ptr, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_parent_ptr, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_new_parent_ptr,size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_pr_arr, pr_size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_label, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_rep, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_OnPath, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_new_OnPath, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_marked_parent,size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_next, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_new_next, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_index_ptr, size), "Failed to allocate memory");
	CUDA_CHECK(cudaMalloc((void **)&d_pr_size_ptr, size), "Failed to allocate memory");

	#ifdef DEBUG
		std::vector<int> rep(n),par(n),marked(n),pr_arr(pr_size),pr_arr_size(n);
	#endif

	int grafting_time = 0, shortcutting_time = 0 , reroot_time = 0;
	int err = 0;
	// Till here pointerJumping values set up

	int numThreads = 1024;
	int numBlocks_n = (vertices + numThreads - 1) / numThreads;

	auto start = std::chrono::high_resolution_clock::now();

	// Step 1: Initialize rep with vertices themselves
	init<<<numBlocks_n, numThreads>>>(d_ptr, d_parent_ptr, vertices);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");


	#ifdef DEBUG
		// std::cout << "Rep array initially : \n";
		// cudaMemcpy(rep.data(), d_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
		// printArr(rep,vertices,10);
	#endif

	int *d_flag;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)), "Failed to allocate memory");

	int flag = 1;
	int iter_number = 0;
	// int numBlocks_e = (edges + numThreads - 1) / numThreads;

	while (flag) {
		if(iter_number > 2*log_2_size)
		{
			std::cout<<"Iterations exceeded 2*log_2_n : "<<iter_number<<"\n";
			err = 1;
			break;
		}

		#ifdef DEBUG
			std::cout<<"\nIteration : "<<iter_number<<"\n";
		#endif

		#ifdef DEBUG
			// cudaMemcpy(rep.data(), d_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
			// std::cout<<"No of components intially : "<<numberOfComponents(rep)<<"\n";
		#endif

		flag = 0;
		CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy");
		CUDA_CHECK(cudaMemset(d_OnPath, 		 0, size), "Failed to memset");
		CUDA_CHECK(cudaMemset(d_index_ptr,		 0, size), "Failed to memset");
		CUDA_CHECK(cudaMemset(d_marked_parent,	-1, size), "Failed to memset");
		
		//thrust::fill is better optimized than cudaMemset
		thrust::fill(d_winner.begin(),d_winner.end(), -1);

		//Step 2: Graft

		auto start_graft = std::chrono::high_resolution_clock::now();

		Graft(vertices, edges, d_u_arr, d_v_arr, d_ptr, d_winner_ptr, d_marked_parent, d_OnPath, d_flag);
		
		auto end_graft = std::chrono::high_resolution_clock::now();
		auto duration_graft = std::chrono::duration_cast<std::chrono::milliseconds>(end_graft - start_graft).count();
	
		grafting_time += duration_graft;
	
		#ifdef DEBUG
			// cudaMemcpy(marked.data(), d_marked_parent, sizeof(int) * n, cudaMemcpyDeviceToHost);
			// std::cout<<"No of marked components : "<<markedComponents(marked)<<"\n";
		#endif
		
		#ifdef DEBUG
    		std::cout << "Marked parent array :\n";
			for(int i=0;i<n;i++)
			{
				if(marked[i] != -1)
					std::cout<< i << " : " << marked[i] << "\n";
			}
		#endif

		// Step 3: ReRoot
		auto start_reroot = std::chrono::high_resolution_clock::now();

		ReRoot(vertices, edges, log_2_size, iter_number, d_OnPath, d_new_OnPath , d_pr_arr, d_parent_ptr, d_new_parent_ptr, d_index_ptr, d_pr_size_ptr, d_marked_parent, d_ptr);

		auto end_reroot = std::chrono::high_resolution_clock::now();
		auto duration_reroot = std::chrono::duration_cast<std::chrono::milliseconds>(end_reroot - start_reroot).count();
	
		reroot_time += duration_reroot;
		
		// #ifdef DEBUG		
		// 	cudaMemcpy(par.data(), d_parent_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);		
		// 	std::cout<<"No of roots after rerooting : "<<rootedComponents(par)<<"\n";
		// #endif
		
		cudaMemcpy(d_next, d_parent_ptr, size, cudaMemcpyDeviceToDevice);

		#ifdef DEBUG
    		// std::cout << "Parent array after rerooting : ";
			// cudaMemcpy(par.data(), d_parent_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
			// printArr(par,vertices,10);
		#endif

		#ifdef DEBUG
	    	// std::cout <<"Rep array before shortcutting: ";
			// cudaMemcpy(rep.data(), d_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
			// printArr(rep,vertices,10);
	    #endif

		// Step 4: Shortcutting
		CUDA_CHECK(cudaMemset(d_pr_size_ptr,0,size), "Failed to memset");
		CUDA_CHECK(cudaMemset(d_pr_arr, -1, pr_size), "Failed to memset");

		auto start_shortcut = std::chrono::high_resolution_clock::now();
		
		Shortcut(vertices, edges, log_2_size, d_next, d_new_next, d_pr_arr, d_ptr, d_pr_size_ptr);	
	
		#ifdef DEBUG
			// cudaMemcpy(rep.data(), d_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);		
			// std::cout<<"No of roots after shortcutting: "<<numberOfComponents(rep)<<"\n";	
		#endif

		#ifdef DEBUG
	    	// std::cout <<"Rep array after shortcutting: ";
			// cudaMemcpy(rep.data(), d_ptr, sizeof(int) * n, cudaMemcpyDeviceToHost);
			// printArr(rep,vertices,10);
	    #endif

		// #ifdef DEBUG
	    // 	cudaMemcpy(pr_arr.data(), d_pr_arr, sizeof(int) * pr_size, cudaMemcpyDeviceToHost);
		// 	cudaMemcpy(pr_arr_size.data(), d_pr_size_ptr, size, cudaMemcpyDeviceToHost);
	    // 	printPR(pr_arr,pr_arr_size,vertices,log_2_size);
	    // #endif
		
		iter_number++;
		CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy");
		
		#ifdef DEBUG
			std::cout << "Flag = " << flag << std::endl;
		#endif
		// break;
	}
	
	// cudaMemcpy(d_parent_ptr,d_next_ptr, size, cudaMemcpyDeviceToDevice);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("PR-RST module", duration);

	#ifdef DEBUG
		std::vector<int> h_parent(n);
		CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent_ptr, n*sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back");
		std::cout << "parent array : \n";

		int j = 0;
		for (auto i : h_parent)
			std::cout << "parent[" << j++ << "] = " << i << std::endl;
		std::cout << std::endl;
	#endif
	
	// std::cout << "Number of iterations taken: "<<iter_number <<"\n";
	// std::cout << "Duration in milliseconds: " << duration << "ms\n";
	// std::cout << "Duration in milliseconds for grafting: " << grafting_time << "ms\n";
	// std::cout << "Duration in milliseconds for reroot: " << reroot_time << "ms\n";
	// std::cout << "Duration in milliseconds for shortcutting: " << shortcutting_time << "ms\n";

	cudaFree(d_OnPath);
	
	#ifdef CHECKER
		if(validateRST(h_parent)) {
			std::cout<<"Validation success"<<std::endl;
			std::cout << "tree depth = " << treeDepth(h_parent) << std::endl;
		}
		else {
			std::cerr << "Validation failure" << std::endl;
		}
	#endif

	return d_parent_ptr;
}
