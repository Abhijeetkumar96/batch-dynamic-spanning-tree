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

#include "connected_components/cc.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>

#include "dynamic_spanning_tree/update_ds.cuh"
#include "common/cuda_utility.cuh"

using namespace cub;

// #define DEBUG

CachingDeviceAllocator g_allocator_(true);  // Caching allocator for device memory


__global__ 
void setParentLevelKernel(int* d_parent, int* d_level, int* roots, int root_count) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid < root_count) {
        int root = roots[tid];
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
    int no_of_vertices, 
    long numEdges, 
    long* d_offset, 
    int* d_neighbours, 
    int* d_level, 
    int* d_parent, 
    int* d_roots,
    int root_count) 
{

    #ifdef DEBUG
        g_verbose = true;
    #endif

    int level = 0;
    int totalThreads = 1024;
    int no_of_blocks = (no_of_vertices + totalThreads - 1) / totalThreads;
    
    int* d_changed;
    cudaMallocManaged(&d_changed, sizeof(int));

    *d_changed= 1;

    int no_of_blocks_roots = (root_count + totalThreads - 1) / totalThreads;

    auto start = std::chrono::high_resolution_clock::now();
    setParentLevelKernel<<<no_of_blocks_roots, totalThreads>>>(d_parent, d_level, d_roots, root_count);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to launch setParentLevelKernel.");

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

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("CUDA BFS", duration);

    std::cout << "Depth of tree: " << level << std::endl;

    CUDA_CHECK(cudaFree(d_changed), "Failed to free d_changed");
}

// ====[ End of constructSpanningTree Code ]====


__global__
void get_original_edges(uint64_t* d_edgeList, int* original_u, int* original_v, long numEdges) {
	
	long tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numEdges) { 
        uint64_t t = d_edgeList[tid];
        original_u[tid] = (int)(t >> 32);
        original_v[tid] = (int)t & 0xFFFFFFFF;
    }
}

__global__
void print_original_edges(int* original_u, int* original_v, long numEdges) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        printf("Printing edgelist from bfs:\n");
        for(long i = 0; i < numEdges; ++i) {
            printf("edge[%ld]: (%d, %d)\n", i, original_u[i], original_v[i]);
        }
    }
}

void print_CSR(const std::vector<long>& vertices, const std::vector<int>& edges) {
    int numVertices = vertices.size() - 1;
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
        }
        std::cout << "\n";
    }
}

int* cuda_BFS(int* original_u, int* original_v, int numVert, long numEdges) {

    auto CC_out = cc(original_u, original_v, numVert, numEdges);
    
    int* d_roots = CC_out.first;
    int root_count = CC_out.second;
    
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

    CUDA_CHECK(cudaMemset(d_level, -1, numVert * sizeof(int)), "Failed to initialize level array.");

    // Step [i]: alternate buffers for sorting operation
    // Create DoubleBuffers
    cub::DoubleBuffer<int> d_u_arr(u_arr_buf, u_arr_alt_buf);
    cub::DoubleBuffer<int> d_v_arr(v_arr_buf, v_arr_alt_buf);

    auto start = std::chrono::high_resolution_clock::now();
	create_duplicate(original_u, original_v, u_arr_buf, v_arr_buf, numEdges);

	// Output: 
	// Vertices array			-> d_vertices <- type: long;
	// Neighbour/edges array	-> d_v_arr.Current() <- type: int;

	gpu_csr(d_u_arr, d_v_arr, E, numVert, d_vertices);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("GPU_CSR", duration);

	// CSR creation ends here

    // if(g_verbose) {
    //     // print gpu_CSR
    //     size_t size = E * sizeof(int);
    //     std::vector<long> host_vert(numVert + 1);
    //     std::vector<int> host_edges(E);
    //     CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream before cudaMemcpyAsync in gpu_csr");
    //     // Use cudaMemcpyAsync with the stream for asynchronous memory copy
    //     CUDA_CHECK(cudaMemcpy(host_vert.data(), d_vertices, (numVert + 1) * sizeof(long), cudaMemcpyDeviceToHost), 
    //                 "Failed to copy back vertices array.");
    //     CUDA_CHECK(cudaMemcpy(host_edges.data(), d_v_arr.Current(), size, cudaMemcpyDeviceToHost), 
    //                 "Failed to copy back edges array.");

    //     print_CSR(host_vert, host_edges);
    // }

    // std::cout << "d_level array from cuda_BFS:\n";
    // print_device_array(d_level, numVert);

	// Step 1: Construct a rooted spanning tree
	constructSpanningTree(
		numVert, 
		E, 
		d_vertices, 
		d_v_arr.Current(), 
		d_level, 
		d_parent, 
		d_roots,
        root_count);

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

    return d_parent;
}