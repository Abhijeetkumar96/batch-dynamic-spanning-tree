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

#include "common/Timer.hpp"

#include "PR-RST/rootedSpanningTreePR.cuh"
#include "dynamic_spanning_tree/update_ds.cuh"
#include "common/cuda_utility.cuh"

using namespace cub;

// #define DEBUG
#define CHECKER

CachingDeviceAllocator g_allocator_(true);  // Caching allocator for device memory

__device__ __forceinline__
long binary_search(uint64_t* array, long num_elements, uint64_t key) {
    long left = 0;
    long right = num_elements - 1;
    while (left <= right) {
        long mid = left + (right - left) / 2;
        if (array[mid] == key) {
            return mid; // Key found
        }
        if (array[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // Key not found
}

__global__
void mark_delete_edges_kernel(
    uint64_t* d_edge_list,  // size <- numEdges
    long num_edges,
    uint64_t* d_edges_to_delete, // size <- delete_batch_size
    int delete_batch_size, 
    unsigned char* d_flags)     // size <- numEdges
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < delete_batch_size) {

        uint64_t t = d_edges_to_delete[tid];

        // t is the key, to be searched in the d_edge_list array
        long pos = binary_search(d_edge_list, num_edges, t);
        
        if(pos != -1) {
            d_flags[pos] = 0;
        }
        if (pos != -1 && pos > num_edges) {
            printf("Error in mark_delete_edges_kernel.\n");
}

    }
}

void sort_array_uint64_t_(uint64_t* d_data, long num_items) {
    // Allocate temporary storage for sorting
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free d_temp_storage");
}

void select_flagged_(uint64_t* d_in, uint64_t* d_out, unsigned char* d_flags, long& num_items) {

    long *d_num_selected_out = NULL;
    g_allocator_.DeviceAllocate((void**)&d_num_selected_out, sizeof(long));

    // Allocate temporary storage
    void    *d_temp_storage = NULL;
    size_t  temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator_.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    long h_num;
    CUDA_CHECK(cudaMemcpy(&h_num, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost),
        "Failed to copy back h_num");

    num_items = h_num;
}

void update_edgelist_bfs(
    uint64_t* d_edge_list, uint64_t* host_edge_list, 
    uint64_t* d_updated_ed_list, 
    long& num_edges, 
    uint64_t* d_edges_to_delete, 
    int delete_size) {

    // sort the input edges
    sort_array_uint64_t_(d_edge_list, num_edges);
    
    // init d_flag with true values
    unsigned char   *d_flags = NULL;

    // std::vector<unsigned char> h_flags(num_edges, 1);
    CUDA_CHECK(cudaMalloc((void**)&d_flags, sizeof(unsigned char) * num_edges), 
        "Failed to allocate memory for d_flags");
    CUDA_CHECK(cudaMemset(d_flags, 1, sizeof(unsigned char) * num_edges), "Failed to memset d_flags");

    // CUDA_CHECK(cudaMemcpy(d_flags, h_flags.data(), sizeof(unsigned char) * num_edges, cudaMemcpyHostToDevice),
    //     "Failed to copy back d_flags");

    std::cout << "Delete batch Size: " << delete_size << std::endl;
    int numThreads = 1024;
    int numBlocks = (delete_size + numThreads - 1) / numThreads;

    auto start = std::chrono::high_resolution_clock::now();
    // Launch kernel to mark batch edges for deletion in the actual edge_list
    mark_delete_edges_kernel<<<numBlocks, numThreads>>>(
        d_edge_list, 
        num_edges, 
        d_edges_to_delete, 
        delete_size, 
        d_flags
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after mark_delete_edges_kernel");

    // now delete the edges from the graph array
    select_flagged_(d_edge_list, d_updated_ed_list, d_flags, num_edges);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("update datastr", duration);
}

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

int constructSpanningTree(
    int no_of_vertices, 
    long numEdges, 
    long* d_offset, 
    int* d_neighbours, 
    int* d_level, 
    int* d_parent, 
    int root) 
{

    int level = 0;
    int totalThreads = 1024;
    int no_of_blocks = (no_of_vertices + totalThreads - 1) / totalThreads;
    
    int* d_changed;
    cudaMallocManaged(&d_changed, sizeof(int));

    *d_changed= 1;

    setParentLevelKernel<<<1, 1>>>(d_parent, d_level, root);
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
    return level;
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

void read_delete_batch(const std::string& delete_filename, std::vector<uint64_t>& edges_to_delete) {

    std::ifstream inputFile(delete_filename);
    if (!inputFile) {
        std::cerr << "Failed to open file: " << delete_filename << std::endl;
        return;
    }
    
    // n_edges: Number of edges to delete, including both tree and non-tree edges.
    int n_edges;
    inputFile >> n_edges;

    uint32_t u, v;
    edges_to_delete.reserve(n_edges);

    for (int i = 0; i < n_edges; ++i) {
        inputFile >> u >> v;
        if(u > v) {
            // Ensures u is always less than v for consistent edge representation
            std::swap(u, v);
        }

        edges_to_delete.push_back((uint64_t)(u) << 32 | v);
    }
}

void cuda_BFS(graph& G, const std::string& delete_filename) {

    int numVert     =   G.numVert;
    long numEdges   =   G.numEdges / 2;
    
    // delete the edges
    std::vector<uint64_t> edges_to_delete;
    read_delete_batch(delete_filename, edges_to_delete);

    uint64_t* d_edge_list = nullptr;
    uint64_t* d_updated_ed_list = nullptr;
    uint64_t* d_edges_to_delete = nullptr;

    int num_edges_to_delete = edges_to_delete.size(); 
    size_t delete_size = edges_to_delete.size() * sizeof(uint64_t);
    size_t edge_list_size = G.edge_list.size() * sizeof(uint64_t);

    // if(g_verbose) {
    //     std::cout << "Edge list from cuda_BFS:\n";
    //     for(auto i : G.edge_list) 
    //         std::cout << (i >> 32) <<" " << (i & 0xFFFFFFFF) << " <- " << i << "\n";
    //     std::cout << std::endl;
    // }
    
    CUDA_CHECK(cudaMalloc(&d_edge_list, edge_list_size), "Failed to allocate memory for input edge list");
    CUDA_CHECK(cudaMalloc(&d_updated_ed_list, edge_list_size), "Failed to allocate memory for input edge list");
    CUDA_CHECK(cudaMalloc(&d_edges_to_delete, delete_size), "Failed to allocate memory for edges to delete");

    CUDA_CHECK(cudaMemcpy(d_edge_list, G.edge_list.data(), edge_list_size, cudaMemcpyHostToDevice), "Failed to copy edge list to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_delete, edges_to_delete.data(), delete_size, cudaMemcpyHostToDevice), "Failed to copy edges to delete to device");

	int* original_u;  // single edges
	int* original_v;

	cudaMalloc((void **)&original_u, numEdges * sizeof(int));
    cudaMalloc((void **)&original_v, numEdges * sizeof(int));

    update_edgelist_bfs(
        d_edge_list, 
        G.edge_list.data(), 
        d_updated_ed_list, 
        numEdges,           // numEdges now contains the updated edgelist count
        d_edges_to_delete, 
        num_edges_to_delete);

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

    int totalThreads = 1024;
    int numBlocks = (numEdges + totalThreads - 1) / totalThreads;

    get_original_edges<<<numBlocks, totalThreads>>>(d_updated_ed_list, original_u, original_v, numEdges);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize get_original_edges");
        
    // call PR_RST (baseline 3)
    RootedSpanningTree(
        original_u, 
        original_v, 
        numVert, 
        numEdges);

    auto start = std::chrono::high_resolution_clock::now();
    int n_comp = cc(original_u, original_v, numVert, numEdges, G.original_filename);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("CC", duration);
    if(n_comp > 1)
        return;

    start = std::chrono::high_resolution_clock::now();
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
	int depth = constructSpanningTree(
		numVert, 
		E, 
		d_vertices, 
		d_v_arr.Current(), 
		d_level, 
		d_parent, 
		root);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("simple BFS", duration);
    
    std::cout << "Depth of graph: " << depth << std::endl;
    
    // call adam_polak bfs (baseline 3)
    adam_polak_bfs(numVert, E, d_vertices, d_v_arr.Current(), u_arr_buf, v_arr_buf);    

}