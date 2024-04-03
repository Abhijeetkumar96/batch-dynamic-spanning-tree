#include "PR-RST/shortcutting.cuh"

//Pointer Jumping Kernel
__global__
void UpdatePR(int *next,int *new_next, int* pr_arr, int log_n, int itr_no, int n,int *pr_size) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {
        int starting_index = log_n * tid;
		if((starting_index + itr_no < log_n*(tid+1)))
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
	int *d_pr_size_ptr
)
{
		int numThreads = 1024;
		int numBlocks_n = (vertices + numThreads - 1) / numThreads;
		// int numBlocks_e = (edges + numThreads - 1) / numThreads;
		
        // Step 4.1: Shortcut PR
	
		for (int j = 0; j < log_2_size; ++j) {
	        UpdatePR<<<numBlocks_n, numThreads>>> (d_next,d_new_next, d_pr_arr, log_2_size, j, vertices, d_pr_size_ptr);
	        cudaDeviceSynchronize();
			int * temp = d_new_next;
			d_new_next = d_next;
			d_next = temp;
		}

		// cudaMemcpy(d_new_next,d_next, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);

		cudaMemcpy(d_ptr,d_next, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);
}