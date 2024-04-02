#include "pr_rst/rootedSpanningTreePR.cuh"
#include "pr_rst/grafting.cuh"
#include "pr_rst/reRoot.cuh"
#include "pr_rst/shortcutting.cuh"

__global__ 
void init(int *arr, int *rep, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
	{
		arr[tid] = tid;
		rep[tid] = tid;
	}
}

__global__ 
void init_1(
	int* d_OnPath, 
	int* d_index_ptr, 
	int* d_marked_parent, 
	int* d_winner, 
	int numElements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        d_OnPath[idx] = 0;
        d_index_ptr[idx] = 0;
        d_marked_parent[idx] = -1;
        d_winner[idx] = -1;
    }
}

void RootedSpanningTree(
	RST_Resource_manager& pr_resources, 
	uint64_t* d_edge_list,
	const int numVert, 
	const int numEdges) {

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

	int *d_winner_ptr = pr_resources.d_winner;
	int *d_ptr = pr_resources.d_ptr;
	int *d_parent_ptr = pr_resources.d_parent_ptr;
	int *d_new_parent_ptr = pr_resources.d_new_parent_ptr; // output
	int *d_pr_arr = pr_resources.d_pr_arr;
	int *d_OnPath = pr_resources.d_OnPath;
	int *d_new_OnPath = pr_resources.d_new_OnPath;
	int *d_marked_parent = pr_resources.d_marked_parent;
	int *d_next = pr_resources.d_next;
	int *d_new_next = pr_resources.d_new_next;
	int *d_index_ptr = pr_resources.d_index_ptr;
	int *d_pr_size_ptr = pr_resources.d_pr_size_ptr;

	// Till here pointerJumping values set up

	int numThreads = 1024;
	int numBlocks_n = (vertices + numThreads - 1) / numThreads;

	auto start = std::chrono::high_resolution_clock::now();

	// Step 1: Initialize rep with vertices themselves
	init<<<numBlocks_n, numThreads>>>(d_ptr, d_parent_ptr, vertices);
	CUDA_CHECK(cudaDeviceSynchronize(), "failed to synchronize after init kernel");

	int *d_flag;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)), "failed to allocate memory for d_flag");

	int flag = 1;
	int iter_number = 0;

	while (flag) {
		if(iter_number > 2*log_2_size) {
			std::cerr<<"Iterations exceeded 2*log_2_n : "<<iter_number<<"\n";
			break;
		}

		flag = 0;

		CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy back d_flag");

		// init_1<<<numBlocks_n, numThreads>>>(d_OnPath, d_index_ptr, d_marked_parent, d_winner_ptr, vertices);

		cudaMemset(d_OnPath, 		 0, size);
		cudaMemset(d_index_ptr,		 0, size);
		cudaMemset(d_marked_parent,	-1, size);
		cudaMemset(d_winner_ptr,	-1, size);
		
		//thrust::fill is better optimized than cudaMemset
		// thrust::fill(d_winner ,d_winner.end(), -1);

		CUDA_CHECK(cudaDeviceSynchronize(), "failed to synchronize after init_1 kernel");

		//Step 2: Graft
		Graft(vertices, edges, d_edge_list, d_ptr, d_winner_ptr, d_marked_parent, d_OnPath, d_flag);

		// Step 3: ReRoot
		ReRoot(vertices, edges, log_2_size, iter_number, d_OnPath, d_new_OnPath , d_pr_arr, d_parent_ptr, d_new_parent_ptr, d_index_ptr, d_pr_size_ptr, d_marked_parent, d_ptr);

		CUDA_CHECK(cudaMemcpy(d_next, d_parent_ptr, size, cudaMemcpyDeviceToDevice), "Failed to copy d_next");

		// Step 4: Shortcutting
		CUDA_CHECK(cudaMemset(d_pr_size_ptr, 0, size), " ");
		CUDA_CHECK(cudaMemset(d_pr_arr, -1, pr_size), " ");

		Shortcut(vertices, edges, log_2_size, d_next, d_new_next, d_pr_arr, d_ptr, d_pr_size_ptr);	

		iter_number++;
		CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back d_flag");
	}

	// return d_new_parent_ptr;
}
