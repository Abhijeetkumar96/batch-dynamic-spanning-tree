#include "common/cuda_utility.cuh"
#include "PR-RST/shortcutting.cuh"

//Pointer Jumping Kernel
__global__
void UpdatePR(int *next,int *new_next, int* pr_arr, int log_n, int itr_no, int n,int *pr_size) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {
        long long starting_index = log_n * 1LL * tid;
        long long next_index = log_n * 1LL * (tid+1);
		if((starting_index + itr_no) < next_index)
		{
			pr_arr[starting_index + itr_no] = next[tid];
		}
        if(next[tid] != tid)
        {
			new_next[tid] = next[next[tid]];
			if(next[tid] != next[next[tid]])
				pr_size[tid]++;
        }
		else
		{
			new_next[tid] = tid; 
		}
    }
}

void Shortcut(
	int vertices,
	int edges,
	int log_2_size,
	int *d_next,
	int *d_new_next,
	int *d_pr_arr,
	int *d_ptr,
	int *d_pr_size_ptr)
{
		int numThreads = 1024;
		int numBlocks_n = (vertices + numThreads - 1) / numThreads;
		
        // Step 4.1: Shortcut PR
	
		for (int j = 0; j < log_2_size; ++j) {
	        UpdatePR<<<numBlocks_n, numThreads>>> (d_next,d_new_next, d_pr_arr, log_2_size, j, vertices, d_pr_size_ptr);
	        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize UpdatePR array");
			int * temp = d_new_next;
			d_new_next = d_next;
			d_next = temp;
		}

		CUDA_CHECK(cudaMemcpy(d_ptr,d_next, sizeof(int) * vertices, cudaMemcpyDeviceToDevice),
			"Failed to copy d_ptr/d_next array");
}