#include "PR-RST/grafting.cuh"

__global__
void DetermineWinners(uint64_t* d_edgelist, int *rep, int *winner, int edges, int *d_flag) {
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < edges) {
		// for i from 1 to n:
		//for each neighbor j of vertex i:
		//Assuming u as vertex i and v as all neighbours of u

		uint64_t t = d_edgelist[tid];
        uint32_t u = (uint32_t)t & 0xFFFFFFFF;
        uint32_t v = (uint32_t)(t >> 32);

		int rep_u = rep[u], rep_v = rep[v];

		if(rep_u != rep_v) {
			winner[max(rep_u,rep_v)] = tid;
			*d_flag = 1;
		}
	}
}

__global__ 
void UpdateLabels(uint64_t* d_edgelist, int *rep, int *winner, int edges, int *marked_parent, int *onPath)
{

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < edges)
	{
		uint64_t t = d_edgelist[tid];
        uint32_t u = (uint32_t)t & 0xFFFFFFFF;
        uint32_t v = (uint32_t)(t >> 32);

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
	uint64_t* d_edgelist,
	int *d_ptr,
	int *d_winner_ptr,
	int *d_marked_parent,
	int *d_OnPath,
	int *d_flag) {
	
	int numThreads = 1024;
	
	// int numBlocks_n = (vertices + numThreads - 1) / numThreads;
	int numBlocks_e = (edges + numThreads - 1) / numThreads;

	// Step 2.1: Determine potential winners for each vertex
	DetermineWinners<<<numBlocks_e, numThreads>>> (d_edgelist, d_ptr, d_winner_ptr, edges, d_flag);
	cudaDeviceSynchronize();

	// Step 2.2: Update labels based on winners and mark parents
	UpdateLabels<<<numBlocks_e, numThreads>>>(d_edgelist, d_ptr, d_winner_ptr, edges, d_marked_parent, d_OnPath);
	cudaDeviceSynchronize();	
}