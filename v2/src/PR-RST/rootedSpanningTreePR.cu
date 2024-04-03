#include "PR-RST/rootedSpanningTreePR.cuh"
#include "PR-RST/grafting.cuh"
#include "PR-RST/reRoot.cuh"
#include "PR-RST/pr_rst_util.cuh"
#include "PR-RST/shortcutting.cuh"

#include "common/cuda_utility.cuh"

// #define DEBUG

__global__ 
void init_arrays(int* d_OnPath, int* d_index_ptr, int* d_marked_parent, int* d_winner_ptr, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        d_OnPath[idx] = 0;
        d_index_ptr[idx] = 0;

        d_marked_parent[idx] = -1;
        d_winner_ptr[idx] = -1;
    }
}

void RootedSpanningTree(uint64_t* d_edgelist, int edges, PR_RST& mem_mag) {

	int n = mem_mag.num_vert;
	int vertices = n;

	// Update values for pointerJumping
	int log_2_size = std::ceil(std::log2(n));
	long long pr_size = std::ceil(n * 1LL * log_2_size);
	long long size = n * 1LL * sizeof(int); // For n vertices

	int *d_winner_ptr 		= 	mem_mag.d_winner_ptr;
	int *d_ptr 				= 	mem_mag.d_ptr;
	int *d_parent_ptr 		= 	mem_mag.d_parent_ptr;
	int *d_new_parent_ptr 	= 	mem_mag.d_new_parent_ptr;
	int *d_pr_arr 			= 	mem_mag.d_pr_arr;
	int *d_OnPath 			= 	mem_mag.d_OnPath;
	int *d_new_OnPath 		= 	mem_mag.d_new_OnPath;
	int *d_marked_parent 	= 	mem_mag.d_marked_parent;
	int *d_next 			=	mem_mag.d_next;
	int *d_new_next 		=	mem_mag.d_new_next;
	int *d_index_ptr 		=	mem_mag.d_index_ptr;
	int *d_pr_size_ptr 		= 	mem_mag.d_pr_size_ptr;
	int *d_flag 			=	mem_mag.d_flag;

	// Till here pointerJumping values set up

	auto start = std::chrono::high_resolution_clock::now();

	int flag = 1;
	int iter_number = 0;

	while (flag) {
		if(iter_number > 2*log_2_size) {
			std::cerr<<"Iterations exceeded 2*log_2_n : "<<iter_number<<"\n";
			break;
		}

		flag = 0;

		CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice), 	"Failed to copy flag to device");
	
		int threadsPerBlock = 1024;
		size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
		init_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_OnPath, d_index_ptr, d_marked_parent, d_winner_ptr, n);
		CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after init_arrays kernel");

		//Step 2: Graft
		Graft(vertices, edges, d_edgelist, d_ptr, d_winner_ptr, d_marked_parent, d_OnPath, d_flag);
		ReRoot(vertices, edges, log_2_size, iter_number, d_OnPath, d_new_OnPath , d_pr_arr, d_parent_ptr, d_new_parent_ptr, d_index_ptr, d_pr_size_ptr, d_marked_parent, d_ptr);
		cudaMemcpy(d_next, d_parent_ptr, size, cudaMemcpyDeviceToDevice);

		// Step 4: Shortcutting
		cudaMemset(d_pr_size_ptr, 0, size);
		cudaMemset(d_pr_arr, -1, pr_size);

		Shortcut(vertices, edges, log_2_size, d_next, d_new_next, d_pr_arr, d_ptr, d_pr_size_ptr);	
		
		iter_number++;
		cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	#ifdef DEBUG
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
	#endif
}
