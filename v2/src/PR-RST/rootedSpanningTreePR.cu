#include "PR-RST/rootedSpanningTreePR.cuh"
#include "PR-RST/grafting.cuh"
#include "PR-RST/reRoot.cuh"
#include "PR-RST/shortcutting.cuh"

#include "common/cuda_utility.cuh"

__global__ 
void init(int *arr, int *rep, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
	{
		arr[tid] = tid;
		rep[tid] = tid;
	}
}

void RootedSpanningTree(uint64_t* d_edgelist, const int numVert, const int numEdges) {

	int n = numVert;
	int vertices = n;
	int edges = numEdges;

	std::cout << "No. of vertices = " << vertices << std::endl;
	std::cout << "No. of edges = " << edges << std::endl;

	// Update values for pointerJumping
	std::cout << "log2(n) = " << std::log2(n) << std::endl;
	int log_2_size = std::ceil(std::log2(n));
	long long pr_size = std::ceil(n * 1LL * log_2_size);
	std::cout << "pr_size = " << pr_size << std::endl;
	
	long long size = n * 1LL * sizeof(int); // For n vertices

	std::cout << "size: " <<  size << std::endl;

	int *d_winner_ptr;
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

	CUDA_CHECK(cudaMalloc((void**)&d_winner_ptr, n * sizeof(int)), 		"Failed to allocate memory for d_winner_ptr");
	CUDA_CHECK(cudaMalloc((void**)&d_ptr, size), 						"Failed to allocate memory for d_ptr");
	CUDA_CHECK(cudaMalloc((void**)&d_parent_ptr, size), 				"Failed to allocate memory for d_parent_ptr");
	CUDA_CHECK(cudaMalloc((void**)&d_new_parent_ptr, size), 			"Failed to allocate memory for d_new_parent_ptr");
	CUDA_CHECK(cudaMalloc((void**)&d_pr_arr, sizeof(int) * pr_size), 	"Failed to allocate memory for d_pr_arr");
	CUDA_CHECK(cudaMalloc((void**)&d_label, size), 						"Failed to allocate memory for d_label");
	CUDA_CHECK(cudaMalloc((void**)&d_rep, size), 						"Failed to allocate memory for d_rep");
	CUDA_CHECK(cudaMalloc((void**)&d_OnPath, size), 					"Failed to allocate memory for d_OnPath");
	CUDA_CHECK(cudaMalloc((void**)&d_new_OnPath, size), 				"Failed to allocate memory for d_new_OnPath");
	CUDA_CHECK(cudaMalloc((void**)&d_marked_parent, size), 				"Failed to allocate memory for d_marked_parent");
	CUDA_CHECK(cudaMalloc((void**)&d_next, size), 						"Failed to allocate memory for d_next");
	CUDA_CHECK(cudaMalloc((void**)&d_new_next, size), 					"Failed to allocate memory for d_new_next");
	CUDA_CHECK(cudaMalloc((void**)&d_index_ptr, size), 					"Failed to allocate memory for d_index_ptr");
	CUDA_CHECK(cudaMalloc((void**)&d_pr_size_ptr, sizeof(int)), 		"Failed to allocate memory for d_pr_size_ptr");

	// Till here pointerJumping values set up

	int numThreads = 1024;
	int numBlocks_n = (vertices + numThreads - 1) / numThreads;

	auto start = std::chrono::high_resolution_clock::now();

	// Step 1: Initialize rep with vertices themselves
	init<<<numBlocks_n, numThreads>>>(d_ptr, d_parent_ptr, vertices);
	cudaDeviceSynchronize();

	int *d_flag;
	cudaMalloc(&d_flag, sizeof(int));

	int flag = 1;
	int iter_number = 0;

	while (flag) {
		if(iter_number > 2*log_2_size) {
			std::cerr<<"Iterations exceeded 2*log_2_n : "<<iter_number<<"\n";
			break;
		}

		flag = 0;
		cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_OnPath, 		 0, size);
		cudaMemset(d_index_ptr,		 0, size);
		cudaMemset(d_marked_parent,	-1, size);
		cudaMemset(d_winner_ptr, 	-1, size);

		//Step 2: Graft
		Graft(vertices, edges, d_edgelist, d_ptr, d_winner_ptr, d_marked_parent, d_OnPath, d_flag);
		ReRoot(vertices, edges, log_2_size, iter_number, d_OnPath, d_new_OnPath , d_pr_arr, d_parent_ptr, d_new_parent_ptr, d_index_ptr, d_pr_size_ptr, d_marked_parent, d_ptr);
		cudaMemcpy(d_next, d_parent_ptr, size, cudaMemcpyDeviceToDevice);

		// Step 4: Shortcutting
		cudaMemset(d_pr_size_ptr,0,size);
		cudaMemset(d_pr_arr, -1, pr_size);

		Shortcut(vertices, edges, log_2_size, d_next, d_new_next, d_pr_arr, d_ptr, d_pr_size_ptr);	
		
		iter_number++;
		cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	std::vector<int> h_parent(n), h_rep(n);
	cudaMemcpy(h_parent.data(), d_parent_ptr, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rep.data(), d_ptr, n*sizeof(int), cudaMemcpyDeviceToHost);
	
	std::cout << "parent array : \n";

	int j = 0;
	for (auto i : h_parent)
		std::cout << "parent[" << j++ << "] = " << i << std::endl;
	std::cout << std::endl;

	std::cout<<"Parent before exiting module \n\n";
	for(auto i : h_parent){
		std::cout<<i<<" ";
	}

	std::cout<<std::endl;

	cudaFree(d_OnPath);
}
