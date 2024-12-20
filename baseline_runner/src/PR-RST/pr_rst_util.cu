#include "PR-RST/pr_rst_util.cuh"
#include "common/cuda_utility.cuh"

__global__ 
void init_pr(int *arr, int *rep, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) {
		arr[tid] = tid;
		rep[tid] = tid;
	}
}

PR_RST::PR_RST(int num_vert) : num_vert(num_vert) {
    mem_alloc();
    // mem_init();
}

void PR_RST::mem_alloc() {

	int n = num_vert;
	int vertices = n;

	// std::cout << "Printing from PR_RST mem_alloc:\n";
	// std::cout << "No. of vertices = " << vertices << std::endl;

	// Update values for pointerJumping
	// std::cout << "log2(n) = " << std::log2(n) << std::endl;
	log_2_size = std::ceil(std::log2(n));
	pr_size = std::ceil(n * 1LL * log_2_size);
	// std::cout << "pr_size = " << pr_size << std::endl;

	long long size = n * 1LL * sizeof(int); // For n vertices

	// std::cout << "size: " <<  size << std::endl;

	// CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)), 						"Failed to allocate memory for d_flag");
	// CUDA_CHECK(cudaMalloc((void**)&d_winner_ptr, n * sizeof(int)), 		"Failed to allocate memory for d_winner_ptr");
	// CUDA_CHECK(cudaMalloc((void**)&d_ptr, size), 						"Failed to allocate memory for d_ptr");
	// CUDA_CHECK(cudaMalloc((void**)&d_parent_ptr, size), 				"Failed to allocate memory for d_parent_ptr");
	// CUDA_CHECK(cudaMalloc((void**)&d_new_parent_ptr, size), 			"Failed to allocate memory for d_new_parent_ptr");

	// CUDA_CHECK(cudaMalloc((void**)&d_label, size), 						"Failed to allocate memory for d_label");
	// CUDA_CHECK(cudaMalloc((void**)&d_rep, size), 						"Failed to allocate memory for d_rep");

	CUDA_CHECK(cudaMalloc((void**)&interval, size), 					"Failed to allocate memory for interval");

	CUDA_CHECK(cudaMalloc((void**)&d_edge_u, size), 					"Failed to allocate memory for d_edge_u");
	CUDA_CHECK(cudaMalloc((void**)&d_parent_u, size), 					"Failed to allocate memory for d_parent_u");

	// CUDA_CHECK(cudaMalloc((void**)&d_pr_arr, sizeof(int) * pr_size), 	"Failed to allocate memory for d_pr_arr");
	// CUDA_CHECK(cudaMalloc((void**)&d_OnPath, size), 					"Failed to allocate memory for d_OnPath");
	// CUDA_CHECK(cudaMalloc((void**)&d_new_OnPath, size), 				"Failed to allocate memory for d_new_OnPath");
	// CUDA_CHECK(cudaMalloc((void**)&d_marked_parent, size), 				"Failed to allocate memory for d_marked_parent");
	// CUDA_CHECK(cudaMalloc((void**)&d_next, size), 						"Failed to allocate memory for d_next");
	// CUDA_CHECK(cudaMalloc((void**)&d_new_next, size), 					"Failed to allocate memory for d_new_next");
	// CUDA_CHECK(cudaMalloc((void**)&d_index_ptr, size), 					"Failed to allocate memory for d_index_ptr");
	// CUDA_CHECK(cudaMalloc((void**)&d_pr_size_ptr, sizeof(int)), 		"Failed to allocate memory for d_pr_size_ptr");
}

// void PR_RST::mem_init() {
// 	CUDA_CHECK(cudaMemset(d_winner_ptr, 0, num_vert * sizeof(int)),			"Failed to set d_winner_ptr");
// 	int numThreads = 1024;
// 	int numBlocks_n = (num_vert + numThreads - 1) / numThreads;
// 	// Step 1: Initialize rep with vertices themselves
// 	init_pr<<<numBlocks_n, numThreads>>>(d_ptr, d_parent_ptr, num_vert);
// 	cudaDeviceSynchronize();
// }

PR_RST::~PR_RST() {
    
    // Free allocated device memory
    // CUDA_CHECK(cudaFree(d_winner_ptr), 		"Failed to free d_winner_ptr");
    // CUDA_CHECK(cudaFree(d_ptr), 			"Failed to free d_ptr");
    // CUDA_CHECK(cudaFree(d_parent_ptr), 		"Failed to free d_parent_ptr");
    // CUDA_CHECK(cudaFree(d_new_parent_ptr), 	"Failed to free d_new_parent_ptr");
    // CUDA_CHECK(cudaFree(d_pr_arr), 			"Failed to free d_pr_arr");
    // CUDA_CHECK(cudaFree(d_OnPath), 			"Failed to free d_OnPath");
    // CUDA_CHECK(cudaFree(d_new_OnPath), 		"Failed to free d_new_OnPath");
    // CUDA_CHECK(cudaFree(d_marked_parent), 	"Failed to free d_marked_parent");
    // CUDA_CHECK(cudaFree(d_next), 			"Failed to free d_next");
    // CUDA_CHECK(cudaFree(d_new_next), 		"Failed to free d_new_next");
    // CUDA_CHECK(cudaFree(d_index_ptr), 		"Failed to free d_index_ptr");
    // CUDA_CHECK(cudaFree(d_pr_size_ptr), 	"Failed to free d_pr_size_ptr");
    // CUDA_CHECK(cudaFree(d_flag), 			"Failed to free d_flag");
}
