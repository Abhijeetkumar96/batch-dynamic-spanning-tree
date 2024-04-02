#include "pr_rst/reversePaths.cuh"

__global__
void Reverse(int* onPath, int *parent,int *new_parent,int n) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		if(onPath[tid])
		{
			if(parent[tid] != tid)
				new_parent[parent[tid]] = tid;
		}
	}
}

__global__
void MarkOnPath(int* onPath,int* newOnPath, int* pr_arr,int n, int log_n,int *iter_no) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		if(onPath[tid]) {
            int j = log_n * tid + iter_no[tid];
			if(j < log_n * (tid + 1) && j >= log_n*tid)
			{
				if(pr_arr[j]!=-1)
				{
					newOnPath[pr_arr[j]] = 1;
					iter_no[pr_arr[j]] = iter_no[tid];
				}
			}
        }
	}
}

__global__ 
void DecrementIter(int n,int * onPath,int *iter_no)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		iter_no[tid] = max(0,iter_no[tid]-1);
	}
}

void ReversePaths(
	int vertices,
	int edges,
	int log_2_size,
	int *d_OnPath,
	int *d_new_OnPath,
	int *d_pr_arr,
	int *d_parent_ptr,
	int *d_new_parent_ptr,
	int *d_index_ptr,
	int *d_pr_size_ptr) {

	int numThreads = 1024;
	int numBlocks_n = (vertices + numThreads - 1) / numThreads;
	// int numBlocks_e = (edges + numThreads - 1) / numThreads;

	cudaMemcpy(d_new_OnPath,d_OnPath, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);	

	// Step 3.1: Mark OnPath array, OnPath[u...v] = {1,...,1} denotes path from u to v needs to be reversed
	for (int j = 0; j < log_2_size ; ++j) {
		
		DecrementIter<<<numBlocks_n, numThreads>>> (vertices ,d_OnPath, d_pr_size_ptr);
		cudaDeviceSynchronize();
		
		MarkOnPath<<<numBlocks_n, numThreads>>> (d_OnPath,d_new_OnPath, d_pr_arr, vertices, log_2_size,d_pr_size_ptr);
		cudaDeviceSynchronize();

		cudaMemcpy(d_OnPath,d_new_OnPath, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);

	}

	// // Step 3.2: Reverse the marked paths

	cudaMemcpy(d_new_parent_ptr,d_parent_ptr, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);
	
	Reverse<<<numBlocks_n, numThreads>>> (d_OnPath,d_parent_ptr, d_new_parent_ptr,vertices);
	cudaDeviceSynchronize();

	cudaMemcpy(d_parent_ptr,d_new_parent_ptr, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);
}
