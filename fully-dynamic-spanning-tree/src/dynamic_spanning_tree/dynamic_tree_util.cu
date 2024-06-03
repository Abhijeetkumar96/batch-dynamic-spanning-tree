#include <set>
#include <vector>
#include <numeric> // For std::accumulate
#include <iostream>

#include <cuda_runtime.h>
#include <thrust/sequence.h>

#include "common/cuda_utility.cuh"
#include "dynamic_spanning_tree/update_ds.cuh"
#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "eulerian_tour/disconnected/list_ranking.cuh"
#include "hash_table/HashTable.cuh"

// #define DEBUG

using namespace cub;

void cpu_pointer_jumping(int* parent, int num_vert) {
    for (int i = 0; i < num_vert; ++i) {
        int root = i;
        // Find the root of the current element
        while (parent[root] != root) {
            root = parent[root];
        }
        // Path compression: update the parent of all elements along the path to point directly to the root
        int current = i;
        while (parent[current] != root) {
            int next = parent[current];
            parent[current] = root;
            current = next;
        }
    }
}

// Constructor
dynamic_tree_manager::dynamic_tree_manager(std::vector<int>& parent, const std::string& delete_filename, const std::vector<uint64_t>& edge_list, int _root) {

    num_vert = parent.size();
    num_edges = edge_list.size();
    root = _root;

    parent_array = new int[num_vert]; // Allocate memory for the array

    // Copy data from the input vector to the newly allocated array
    std::memcpy(parent_array, parent.data(), num_vert * sizeof(int));
    // std::cout << "Reading delete edges file\n";
    read_delete_batch(delete_filename, parent);
    // std::cout << "Reading completed.\n";
    
    // std::cout << "Allocating gpu memory\n";
    mem_alloc(parent, edge_list);
    // std::cout << "Allocation over.\n";

    // std::cout << "Updating data structure\n";
    update_existing_ds();
    std::cout << std::endl;
}

size_t AllocateTempStorage(void** d_temp_storage, long num_items) {
    size_t temp_storage_bytes = 0;
    size_t required_bytes = 0;

    // Determine the temporary storage requirement for DeviceRadixSort::SortPairs
    cub::DeviceRadixSort::SortPairs(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceScan::InclusiveSum
    cub::DeviceScan::InclusiveSum(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceSelect::Flagged
    cub::DeviceSelect::Flagged(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Allocate the maximum required temporary storage
    CUDA_CHECK(cudaMalloc(d_temp_storage, temp_storage_bytes), "cudaMalloc failed for temporary storage for CUB operations");

    return temp_storage_bytes;
}

void dynamic_tree_manager::create_hashtable_() {
    pHashTable = create_hashtable();
}

void dynamic_tree_manager::destroy_hashtable_() {
    destroy_hashtable(pHashTable);
}

void dynamic_tree_manager::mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list) {

    size_t size = parent.size() * sizeof(int);
    size_t delete_size = edges_to_delete.size() * sizeof(uint64_t);
    size_t edge_list_size = edge_list.size() * sizeof(uint64_t);

    // std::cout << "size: " << size << std::endl;
    // std::cout << "delete_size: " << delete_size << std::endl;
    // std::cout << "edge_list_size: " << edge_list_size << std::endl;

    // Allocate device memory

    pHashTable = create_hashtable();

    // allocate temp storage
    AllocateTempStorage(&d_temp_storage, 2 * edge_list.size());

    CUDA_CHECK(cudaMalloc(&d_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&new_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&d_org_parent, size), "Failed to allocate memory for d_org_parent");
    CUDA_CHECK(cudaMalloc(&d_unique_rep, size), "Failed to allocate memory for d_unique_rep");
    CUDA_CHECK(cudaMalloc(&d_rep_map, size), "Failed to allocate memory for d_rep_map");
    CUDA_CHECK(cudaMalloc(&d_componentParent, size), "Failed to allocate memory for d_componentParent");
    CUDA_CHECK(cudaMalloc(&d_parentEdge, size), "Failed to allocate memory for d_parentEdge");

    CUDA_CHECK(cudaMalloc(&d_actual_roots, size), "Failed to allocate memory for d_actual_roots");
    CUDA_CHECK(cudaMalloc(&d_super_tree_roots, size), "Failed to allocate memory for d_super_tree_roots");

    CUDA_CHECK(cudaMalloc(&d_roots_flag, num_vert * sizeof(unsigned char)), "Failed to allocate memory for d_roots_flag");

    CUDA_CHECK(cudaMalloc(&d_edges_to_delete, delete_size), "Failed to allocate memory for edges to delete");
    CUDA_CHECK(cudaMalloc(&d_edges_to_insert, delete_size), "Failed to allocate memory for edges to insert");
    CUDA_CHECK(cudaMalloc(&d_super_tree_list, delete_size), "Failed to allocate memory for edges to insert");

    // d_edge_list is the original edge_list
    CUDA_CHECK(cudaMalloc(&d_edge_list, edge_list_size), "Failed to allocate memory for input edge list");
    
    // d_updated_edge_list is the new edgelist after deleting the edges
    CUDA_CHECK(cudaMalloc(&d_updated_edge_list, edge_list_size), "Failed to allocate memory for input edge list");

    CUDA_CHECK(cudaMalloc(&d_cross_edges_flag, num_edges * sizeof(unsigned char)), "Failed to allocate memory for d_cross_edges");

    CUDA_CHECK(cudaMalloc(&d_super_graph_u, num_edges * sizeof(int)), "Failed to allocate device memory for d_super_graph_u");
    CUDA_CHECK(cudaMalloc(&d_super_graph_v, num_edges * sizeof(int)), "Failed to allocate device memory for d_super_graph_v");

    CUDA_CHECK(cudaMalloc(&d_new_super_graph_u, num_edges * sizeof(int)), "Failed to allocate device memory for d_new_super_graph_u");
    CUDA_CHECK(cudaMalloc(&d_new_super_graph_v, num_edges * sizeof(int)), "Failed to allocate device memory for d_new_super_graph_v");

    CUDA_CHECK(cudaMallocManaged(&super_graph_edges, sizeof(int)),   "Failed to allocate d_num_selected_out");
    CUDA_CHECK(cudaMalloc((void**)&d_flags, num_edges * sizeof(unsigned char)), "Failed to allocate flag array");
    
    CUDA_CHECK(cudaMemset(d_componentParent, -1, size), "Failed to memset d_componentParent");
    CUDA_CHECK(cudaMemset(d_parentEdge, -1, size), "Failed to memset d_parentEdge");
    CUDA_CHECK(cudaMemset(d_roots_flag,       0, num_vert * sizeof(unsigned char)), "Failed to memset d_roots");
    CUDA_CHECK(cudaMemset(d_cross_edges_flag, 0, num_edges * sizeof(unsigned char)), "Failed to memset d_cross_edges");
    
    // Fill the array with increasing values starting from 0
    thrust::sequence(thrust::device, d_actual_roots, d_actual_roots + num_vert);

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), size, cudaMemcpyHostToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_org_parent, d_parent, size,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_delete, edges_to_delete.data(), delete_size, cudaMemcpyHostToDevice), "Failed to copy edges to delete to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_insert, d_edges_to_delete, delete_size, cudaMemcpyDeviceToDevice), "Failed to copy edges to delete to device");
    CUDA_CHECK(cudaMemcpy(d_edge_list, edge_list.data(), edge_list_size, cudaMemcpyHostToDevice), "Failed to copy edge list to device");
}

void dynamic_tree_manager::read_delete_batch(const std::string& delete_filename, std::vector<int>& parent) {

    std::ifstream inputFile(delete_filename);
    if (!inputFile) {
        std::cerr << "Failed to open file: " << delete_filename << std::endl;
        return;
    }
    
    // n_edges: Number of edges to delete, including both tree and non-tree edges.
    int n_edges;
    inputFile >> n_edges;
    delete_batch_size = n_edges;
    uint32_t u, v;
    edges_to_delete.resize(n_edges);
    
    tree_edge_count = 0;
    
    // std::cout << "Reading " << n_edges << " edges from the file." << std::endl;

    for (int i = 0; i < n_edges; ++i) {
        inputFile >> u >> v;
        if(u > v) {
            // Ensures u is always less than v for consistent edge representation
            std::swap(u, v);
        }

        if(parent_array[u] == v or parent_array[v] == u) {
            tree_edge_count++;

            if(u == parent_array[v]) {
                parent_array[v] = v; // Disconnect the child from its parent

            } else if (v == parent_array[u]) {
                parent_array[u] = u;
            }
        }

        edges_to_delete[i] = ((uint64_t)(u) << 32 | v);
    }
    cpu_pointer_jumping(parent_array, num_vert);

    std::cout << "Number of deleted tree edges: " << tree_edge_count << std::endl;
    
    if(g_verbose) {

        // std::cout << "edges_to_delete array uint64_t:\n";

        // for(auto i : edges_to_delete)
        //     std::cout << i <<" ";
        // std::cout << std::endl;

        std::cout << "Deleted edges:" << std::endl;
        for(const auto &i : edges_to_delete)
            std::cout << (i >> 32) << ", " << (i & 0xFFFFFFFF) << "\n";
        std::cout << std::endl;
    }
}

void dynamic_tree_manager::update_existing_ds() {
	update_edgelist(
        d_parent,               // input -- 1
        num_vert,               // input -- 2
        d_edge_list,            // input -- 3
        d_updated_edge_list,    // output -- 4
        num_edges,              // output -- 5
        d_edges_to_delete,      // input -- 6
        delete_batch_size,      // input -- 7
        d_unique_rep,           // output -- 8
        root);                  // input -- 10

    // now num_edges contains nonTreeEdges - parent_size - delete_batch count.
}

__global__
void list_ranking_kernel(int* next, int* dist, int* new_next, int* new_dist, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {
        // printf("\nNext array for %d iteration : %d", itr_no, next[tid]);
        // printf("\nDist array : %d", dist[tid]);
        if(next[tid] != tid)
        {
            new_dist[tid] = dist[tid] + dist[next[tid]];
            new_next[tid] = next[next[tid]];
        } else {
          new_dist[tid] = 0;
          new_next[tid] = tid;
        }
    }
}

void listRanking(int* d_next, std::vector<int>& new_dist, int n) {
    std::vector<int> dist(n, 1);  // Initialize distance array with 1
    int size = n * sizeof(int);

    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_dist, sizeof(int) * n), "Failed to allocate d_dist");

    int *d_new_dist, *d_new_next;
    CUDA_CHECK(cudaMalloc((void**)&d_new_dist, size), "Failed to allocate d_new_dist");
    CUDA_CHECK(cudaMalloc((void**)&d_new_next, size), "Failed to allocate d_new_next");

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), size, cudaMemcpyHostToDevice), "Failed to copy dist array");

    // Calculate the optimal number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int j = 0; j < std::ceil(std::log2(n)); ++j) {
        list_ranking_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_dist, d_new_next, d_new_dist, n);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize list_ranking_kernel");

        // Swap pointers
        int* temp = d_new_next;
        d_new_next = d_next;
        d_next = temp;

        temp = d_new_dist;
        d_new_dist = d_dist;
        d_dist = temp;
    }
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after list_ranking_kernel");

    new_dist.resize(n);
    CUDA_CHECK(cudaMemcpy(new_dist.data(), d_dist, size, cudaMemcpyDeviceToHost), "Failed to copy back");
    
    // #ifdef DEBUG
    //     std::cout << "Printing final distance array:\n";
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << "dist[" << j << "] = " << new_dist[j] << std::endl;
    //     }
    //     std::cout << std::endl;
    // #endif

    CUDA_CHECK(cudaFree(d_new_dist), "Failed to free");
    CUDA_CHECK(cudaFree(d_new_next), "Failed to free");
}

void find_path_length(int* d_parent, int num_vert, int& min_rank, int& max_rank, int& avg_rank, int& median_rank) {

    std::vector<int> host_Rank;
    // Call the CUDA function for list ranking
    listRanking(d_parent, host_Rank, num_vert);

    // Extract unique ranks
    std::set<int> unique_ranks(host_Rank.begin(), host_Rank.end());

    std::sort(host_Rank.begin(), host_Rank.end());
    if (num_vert % 2 == 0) {
        median_rank = (host_Rank[num_vert / 2 - 1] + host_Rank[num_vert / 2]) / 2.0;
    } else {
        median_rank = host_Rank[num_vert / 2];
    }

    // Find min, max, and average values
    min_rank = *unique_ranks.begin();
    max_rank = *unique_ranks.rbegin();
    avg_rank = std::accumulate(unique_ranks.begin(), unique_ranks.end(), 0.0) / unique_ranks.size();

    return;
}

__global__
void cross_edges_kernel(
    const uint64_t* d_edge_list,
    int* rep_array, 
    long num_edges,
    int* d_counter) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_edges) {

        // get the actual edges from d_edge_list
        uint64_t edge = d_edge_list[tid];
        
        uint32_t d_nst_u = edge >> 32;
        uint32_t d_nst_v = edge & 0xFFFFFFFF;

        #ifdef DEBUG
            printf("d_u: %d, d_v: %d, rep[u]: %d, rep[v]: %d\n", d_nst_u, d_nst_v, rep_array[d_nst_u], rep_array[d_nst_v]);
        #endif

        if(rep_array[d_nst_u] == rep_array[d_nst_v]) {
            return;
        }

        atomicAdd(d_counter, 1);
    }
}

int cal_cross_edges(int* updated_parent, int num_vert, const uint64_t* d_updated_edge_list, long num_edges) {
    int* d_rep;
    CUDA_CHECK(cudaMalloc(&d_rep, sizeof(int) * num_vert), "Failed to allocate memory for d_rep");
    CUDA_CHECK(cudaMemcpy(d_rep, updated_parent, sizeof(int) * num_vert,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");
    pointer_jumping(d_rep, num_vert);

    #ifdef DEBUG
        std::cout << "After doing pointer_jumping Rep Array:\n";
        print_device_array(d_rep, num_vert);
    #endif

    // find number of cross_edges
    int* d_counter;
    CUDA_CHECK(cudaMallocHost(&d_counter, sizeof(int)), "Failed to allocate d_counter");
    *d_counter = 0;

    long numThreads = 1024;
    long numBlocks = (num_edges + numThreads - 1) / numThreads;

    // create superGraph
    cross_edges_kernel<<<numBlocks, numThreads>>>(
        d_updated_edge_list, 
        d_rep, 
        num_edges,
        d_counter
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    int h_counter = *d_counter;

    cudaFree(d_rep);
    cudaFreeHost(d_counter);

    return h_counter;
}

void dynamic_tree_manager::print_stats() {
    int size = sizeof(int) * num_vert;
    int* updated_parent;
    CUDA_CHECK(cudaMalloc(&updated_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMemcpy(updated_parent, d_parent, size,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");

    int min_length, max_length, avg_length, median_length;
    int cross_edges_count = cal_cross_edges(updated_parent, num_vert, d_updated_edge_list, num_edges);
    find_path_length(updated_parent, num_vert, min_length, max_length, avg_length, median_length);

    std::cout << "\nUpdated edge_list size = " << num_edges;
    std::cout << "\nmax path length = " << max_length;
    std::cout << "\navg path length = " << avg_length;
    std::cout << "\nmedian path length = " << median_length;
    std::cout << "\nnum cross edges = " << cross_edges_count;
    std::cout << "\n----------------------------------------" << std::endl;
}

dynamic_tree_manager::~dynamic_tree_manager() {
    delete[] parent_array;

    cudaFree(d_parent);
    cudaFree(d_unique_rep);
    cudaFree(d_edges_to_delete);
    cudaFree(d_edge_list);
    destroy_hashtable(pHashTable);
}

// ====[ End of update ds Code ]====