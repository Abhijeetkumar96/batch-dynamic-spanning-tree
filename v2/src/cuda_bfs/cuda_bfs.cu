//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

//---------------------------------------------------------------------
// CUDA Libraries
//---------------------------------------------------------------------
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// CUDA Kernels
//---------------------------------------------------------------------
#include "cuda_bfs/cuda_csr.cuh"
#include "cuda_bfs/cuda_bfs.cuh"
#include "common/cuda_utility.cuh"
#include "common/Timer.hpp"


__global__ 
void setParentLevelKernel(int* d_parent, int* d_level, int root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_parent[root] = root;
        d_level[root] = 0;
    }
}

__global__ 
void simpleBFS( 
	int no_of_vertices, int level, 
    int* d_parents, int* d_levels, 
    long* d_offset, int* d_neighbour, 
    int* d_changed) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < no_of_vertices && d_levels[tid] == level) {
        int u = tid;
        for (long i = d_offset[u]; i < d_offset[u + 1]; i++) {
            int v = d_neighbour[i];
            if(d_levels[v] < 0) {
                d_levels[v] = level + 1;
                d_parents[v] = u;
                *d_changed = 1;
            }
        }
    }
}

void constructSpanningTree(
    int no_of_vertices, long numEdges, 
    long* d_offset, int* d_neighbours, 
    int* d_level, int* d_parent, int root) {

    int level = 0;
    int totalThreads = 1024;
    int no_of_blocks = (no_of_vertices + totalThreads - 1) / totalThreads;
    
    int* d_changed;
    cudaMallocManaged(&d_changed, sizeof(int));

    *d_changed= 1;

    setParentLevelKernel<<<1, 1>>>(d_parent, d_level, root);
    CUDA_CHECK(cudaGetLastError(), "Failed to launch setParentLevelKernel.");
    
    while (*d_changed) {
        *d_changed = 0;
        
        simpleBFS<<<no_of_blocks, totalThreads>>>(
            no_of_vertices, 
            level, 
            d_parent, 
            d_level, 
            d_offset, 
            d_neighbours, 
            d_changed
        );
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after simpleBFS");
        ++level;
    }
}

// ====[ End of constructSpanningTree Code ]====


__global__
void get_original_edges(uint64_t* d_edgeList, int* original_u, int* original_v, long numEdges) {
	
	long tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	uint64_t t = d_edgeList[tid];
    original_u[tid] = (int)t & 0xFFFFFFFF;
    original_v[tid] = (int)(t >> 32);
}

void cuda_BFS(uint64_t* d_edgeList, int numVert, int numEdges) {
	
	int* original_u;  // single edges
	int* original_v;

	cudaMalloc((void **)&original_u, numEdges * sizeof(int));
    cudaMalloc((void **)&original_v, numEdges * sizeof(int));

	long E = 2 * numEdges; // Two times the original edges count (0,1) and (1,0).
	// step 1: Create duplicates
	int* u_arr_buf;
	int* v_arr_buf;
	int* u_arr_alt_buf;
	int* v_arr_alt_buf;

	// Allocate memory for duplicates
    cudaMalloc((void **)&u_arr_buf, E * sizeof(int));
    cudaMalloc((void **)&v_arr_buf, E * sizeof(int));
    cudaMalloc((void **)&u_arr_alt_buf, E * sizeof(int));
    cudaMalloc((void **)&v_arr_alt_buf, E * sizeof(int));

    long* d_vertices;
	cudaMalloc((void **)&d_vertices, (numVert + 1) * sizeof(long));

	int *d_parent;
	int *d_level;

	cudaMalloc((void **)&d_parent,  numVert * sizeof(int));
    cudaMalloc((void **)&d_level,   numVert * sizeof(int));

    int totalThreads = 1024;
    int numBlocks = (numEdges + totalThreads - 1) / totalThreads;

    Timer myTimer;
    myTimer.start();
    std::cout << "Timer started" << std::endl;

    get_original_edges<<<numBlocks, totalThreads>>>(d_edgeList, original_u, original_v, numEdges);

	create_duplicate(original_u, original_v, u_arr_buf, v_arr_buf, numEdges);

	// Step [i]: alternate buffers for sorting operation
	// Create DoubleBuffers
	cub::DoubleBuffer<int> d_u_arr(u_arr_buf, u_arr_alt_buf);
	cub::DoubleBuffer<int> d_v_arr(v_arr_buf, v_arr_alt_buf);

	// Output: 
	// Vertices array			-> d_vertices <- type: long;
	// Neighbour/edges array	-> d_v_arr.Current() <- type: int;

	gpu_csr(d_u_arr, d_v_arr, E, numVert, d_vertices);
	// CSR creation ends here
	
	int root = 0;
	// Step 1: Construct a rooted spanning tree
	constructSpanningTree(
		numVert, 
		E, 
		d_vertices, 
		d_v_arr.Current(), 
		d_level, 
		d_parent, 
		root);

    std::cout << "Total elapsed time for cudaBFS: " << myTimer.getElapsedMilliseconds() << " ms" << std::endl;

	// Cleanup
    cudaFree(original_u);
    cudaFree(original_v);
    cudaFree(u_arr_buf);
    cudaFree(v_arr_buf);
    cudaFree(u_arr_alt_buf);
    cudaFree(v_arr_alt_buf);
	cudaFree(d_vertices);
	cudaFree(d_parent);
	cudaFree(d_level);
}