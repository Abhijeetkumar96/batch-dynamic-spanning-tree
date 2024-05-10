#include "PR-RST/reRoot.cuh"
#include "PR-RST/reversePaths.cuh"

__global__
void AssignParents(int *marked_parent,int *parent,int n)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if(tid < n)
	{
		if(marked_parent[tid]!=-1)
		{
			parent[tid] = marked_parent[tid];
		}
	}
}

void ReRoot(
	int vertices,
	int edges,
	int log_2_size,
	int iter_number,
	int *d_OnPath,
	int *d_new_OnPath,
	int *d_pr_arr,
	int *d_parent_ptr,
	int *d_new_parent_ptr,
	int *d_index_ptr,
	int *d_pr_size_ptr,
	int *d_marked_parent,
	int *d_ptr
)
{
		int numThreads = 1024;
		int numBlocks_n = (vertices + numThreads - 1) / numThreads;
		// int numBlocks_e = (edges + numThreads - 1) / numThreads;

		if(iter_number >= 1)
		{
			// Step 3.1: Reverse Paths in each component
			ReversePaths(vertices,edges,log_2_size,d_OnPath,d_new_OnPath,d_pr_arr,d_parent_ptr, d_new_parent_ptr,d_index_ptr,d_pr_size_ptr);
			// #ifdef DEBUG
				
			// #endif
		}

		// Step 3.2: Assign parent child relationship
		AssignParents<<<numBlocks_n,numThreads>>>(d_marked_parent,d_parent_ptr,vertices);
		cudaDeviceSynchronize();

}