#include "PR-RST/grafting.cuh"
#include "common/cuda_utility.cuh"

__global__
void DetermineWinners(int* d_u_arr, int* d_v_arr, int *rep, int *winner, int edges, int *d_flag) {
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < edges) {
		// for i from 1 to n:
		//for each neighbor j of vertex i:
		//Assuming u as vertex i and v as all neighbours of u

        int u = d_u_arr[tid];
        int v = d_v_arr[tid];

		int rep_u = rep[u], rep_v = rep[v];

		if(rep_u != rep_v) {
			winner[max(rep_u,rep_v)] = tid;
			*d_flag = 1;
		}
	}
}

__global__ 
void UpdateLabels(int* d_u_arr, int* d_v_arr, int *rep, int *winner, int edges, int *marked_parent, int *onPath)
{

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < edges)
	{
		int u = d_u_arr[tid];
        int v = d_v_arr[tid];

		int rep_u = rep[u], rep_v = rep[v];

		if(rep_u != rep_v && winner[max(rep_u,rep_v)] == tid)
		{
			if(rep_u > rep_v)
			{
				marked_parent[u] = v;
				onPath[u] = 1;
			}
			else
			{
				marked_parent[v] = u;
				onPath[v] = 1;	
			}

		}
	}
}

void Graft(
	int vertices,
	int edges,
	int* d_u_arr, 
	int* d_v_arr,
	int *d_ptr,
	int *d_winner_ptr,
	int *d_marked_parent,
	int *d_OnPath,
	int *d_flag) {
	
	int numThreads = 1024;
	
	// int numBlocks_n = (vertices + numThreads - 1) / numThreads;
	int numBlocks_e = (edges + numThreads - 1) / numThreads;

	// Step 2.1: Determine potential winners for each vertex
	DetermineWinners<<<numBlocks_e, numThreads>>> (d_u_arr, d_v_arr, d_ptr, d_winner_ptr, edges, d_flag);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

	#ifdef DEBUG
		std::cout << "d_winner_ptr:\n";
		print_device_array(d_winner_ptr, vertices); 
	#endif

	// Step 2.2: Update labels based on winners and mark parents
	UpdateLabels<<<numBlocks_e, numThreads>>>(d_u_arr, d_v_arr, d_ptr, d_winner_ptr, edges, d_marked_parent, d_OnPath);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

	#ifdef DEBUG
		std::cout << "d_marked_parent:\n";
		print_device_array(d_marked_parent, vertices);
		std::cout << "d_OnPath:\n";
		print_device_array(d_OnPath, vertices);
		CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
	#endif
}