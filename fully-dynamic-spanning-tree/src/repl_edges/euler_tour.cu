#include <iostream>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "common/cuda_utility.cuh"
#include "eulerian_tour/disconnected/list_ranking.cuh"
#include "repl_edges/euler_tour.cuh"

// #define DEBUG

__global__ 
void create_dup_edges(
    int *d_edges_to, 
    int *d_edges_from, 
    const int *edges_to_input, 
    const int *edges_from_input, 
    int edge_count) {

    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < edge_count) {
        d_edges_from[thid + edge_count] = d_edges_to[thid] = edges_to_input[thid];
        d_edges_to[thid + edge_count] = d_edges_from[thid] = edges_from_input[thid];
    }
}

__global__
void init_nxt(int* d_next, int E) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < E) {
        d_next[thid] = -1;
    }
}

__global__
void update_first_last_nxt(int* d_edges_from, int* d_edges_to, int* d_first, int* d_last, int* d_next, uint64_t* d_index, int E) {
    
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < E) {
        int f = d_edges_from[d_index[thid]];
        int t = d_edges_to[d_index[thid]];

        if (thid == 0) {
            d_first[f] = d_index[thid];
            return;
        }

        if(thid == E - 1) {
            d_last[f] = d_index[thid];
        }

        int pf = d_edges_from[d_index[thid - 1]];
        int pt = d_edges_to[d_index[thid - 1]];

        // printf("For tid: %d, f: %d, t: %d, pf: %d, pt: %d\n", thid, f, t, pf, pt);

        // calculate the offset array
        if (f != pf) {
            d_first[f] = d_index[thid];
            // printf("d_last[%d] = d_index[%d] = %d\n", pf, thid - 1, d_index[thid - 1]);
            d_last[pf] = d_index[thid - 1];
        } else {
            d_next[d_index[thid - 1]] = d_index[thid];
        }
    }
}

__global__ 
void cal_succ(int* succ, const int* d_next, const int* d_first, const int* d_edges_from, int E) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < E) {
        int revEdge = (thid + E / 2) % E;

        if (d_next[revEdge] == -1) {
            succ[thid] = d_first[d_edges_from[revEdge]];
        } else {
            succ[thid] = d_next[revEdge];
        }
    }
}

__global__ 
void break_cycle_kernel(int *d_last, int *d_succ, int* d_roots, int roots_count, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < roots_count) {
        int root = d_roots[idx];
        // printf("Root: %d\n", root);
        if (d_last[root] != -1) {
            int last_edge = d_last[root];
            int rev_edge = (last_edge + E / 2) % E;
            // printf("\nFor root: %d, last_edge: %d, rev_edge: %d\n", root, last_edge, rev_edge);
            // Set the successor of the last edge to point to itself
            d_succ[rev_edge] = -1;
        }
    }
}


__global__
void find_parent(int E, int *rank, int *d_edges_to, int *d_edges_from, int *parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < E) {
        int f = d_edges_from[tid];
        int t = d_edges_to[tid];
        int rev_edge = (tid + E / 2) % E;
        // printf("for tid: %d, f: %d, t: %d, rev_edge: %d\n", tid, f, t, rev_edge);
        if(rank[tid] > rank[rev_edge]) {
            parent[t] = f;
        }
        else {
            parent[f] = t;
        }
    }
}

__global__ 
void merge_key_value(const int *arrayU, const int *arrayV, uint64_t *arrayE, uint64_t *d_indices, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Cast to int64_t to ensure the shift operates on 64 bits
        uint64_t u = arrayU[idx];
        uint64_t v = arrayV[idx];

        arrayE[idx] = (u << 32) | (v & 0xFFFFFFFFLL);

        d_indices[idx] = idx;
    }
}

void LexSortIndices(int* d_keys, int* d_values, uint64_t* d_indices_sorted, int num_items, uint64_t *d_merged, uint64_t *d_merged_keys_sorted, uint64_t* d_indices) {

    int blockSize = 1024;
    int numBlocks = (num_items + blockSize - 1) / blockSize; 

    // Initialize indices to 0, 1, 2, ..., num_items-1 also here
    merge_key_value<<<numBlocks, blockSize>>>(
        d_keys, 
        d_values, 
        d_merged, 
        d_indices, 
        num_items);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize merge_key_value kernel");

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to synchronize d_temp_storage");
    
    // Sort indices based on keys
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);
}


// CUDA kernel to unmerge an array of int64_t into two integer arrays
__global__ 
void unpackPairs_(const uint64_t *zippedArray, int *arrayA, int *arrayB, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Extract the upper 32 bits
        arrayA[idx] = zippedArray[idx] >> 32;
        // Extract the lower 32 bits, ensuring it's treated as a signed int
        arrayB[idx] = int(zippedArray[idx] & 0xFFFFFFFFLL);  
    }
}

void cuda_euler_tour(uint64_t* d_edge_list, int N, int num_edges, int* d_roots, int roots_count, Euler_Tour& euler) {
    
    #ifdef DEBUG
        std::cout << "num_vert: " << N << std::endl;
        std::cout << "num_edges: " << num_edges << std::endl;

        std::cout << "Edges input to euler: " << std::endl;
        print_device_edge_list(d_edge_list, num_edges);
    #endif

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t* d_sorted_keys = euler.d_sorted_keys;

    int* d_u = euler.d_u;
    int* d_v = euler.d_v;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_edge_list, d_sorted_keys, num_edges);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate d_temp_storage");

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_edge_list, d_sorted_keys, num_edges);

    int blockSize = 1024;
    int numBlocks = (num_edges + blockSize - 1) / blockSize; 

    unpackPairs_<<<numBlocks, blockSize>>>(d_sorted_keys, d_u, d_v, num_edges);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after unpackPairs_ kernel");

    int E = num_edges * 2;
    
    int *d_edges_to = euler.d_edges_to;
    int *d_edges_from = euler.d_edges_from;
    
    // index can be considered as edge_num
    uint64_t *d_index = euler.d_index;

    int *d_next = euler.d_next;

    numBlocks = (N - 1 + blockSize - 1) / blockSize; 

    // Launch the kernel
    create_dup_edges<<<numBlocks, blockSize>>>(
        d_edges_to, 
        d_edges_from, 
        d_u, 
        d_v, 
        E/2);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize create_dup_edges");
    
    #ifdef DEBUG
        std::cout << "Edgelist from Euler:" << std::endl;
        DisplayDeviceEdgeList(d_edges_from, d_edges_to, E);
    #endif

    numBlocks = (E + blockSize - 1) / blockSize;

    init_nxt<<<numBlocks, blockSize>>>(d_next, E);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_nxt kernel"); 
    
    uint64_t *d_merged              = euler.d_merged;
    uint64_t *d_merged_keys_sorted  = euler.d_merged_keys_sorted;
    uint64_t* d_indices             = euler.d_indices;

    LexSortIndices(d_edges_from, d_edges_to, d_index, E, d_merged, d_merged_keys_sorted, d_indices);

    #ifdef DEBUG
        std::cout << "Index array:" << std::endl;
        print_device_array(d_index, E);

        std::vector<int> sorted_from(E), sorted_to(E);
        std::vector<uint64_t> sorted_index(E);
        
        CUDA_CHECK(cudaMemcpy(sorted_index.data(), d_index, sizeof(uint64_t) * E, cudaMemcpyDeviceToHost), "Failed to copy d_index");
        CUDA_CHECK(cudaMemcpy(sorted_from.data(),  d_edges_from, sizeof(int) * E, cudaMemcpyDeviceToHost), "Failed to copy d_edges_from");
        CUDA_CHECK(cudaMemcpy(sorted_to.data(),    d_edges_to,   sizeof(int) * E, cudaMemcpyDeviceToHost), "Failed to copy d_edges_to");

        // Print the sorted edges
        std::cout << "Sorted Edges:" << std::endl;
        for (int i = 0; i < E; ++i) {
            int idx = sorted_index[i];
            std::cout << i << ": (" << sorted_from[idx] << ", " << sorted_to[idx] << ")" << std::endl;
        }
    #endif

    int *d_first = euler.d_first;
    int *d_last = euler.d_last;
    
    update_first_last_nxt<<<numBlocks, blockSize>>>(
        d_edges_from, 
        d_edges_to, 
        d_first, 
        d_last, 
        d_next, 
        d_index, 
        E);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_first_last_nxt kernel");

    int *succ = euler.succ;
    int *devRank = euler.devRank;

    cal_succ<<<numBlocks, blockSize>>>(succ, d_next, d_first, d_edges_from, E);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize call_succ kenrel");

    #ifdef DEBUG
        std::cout << "successor array before break_cycle_kernel:" << std::endl;
        print_device_array(succ, E);
    #endif

    // break cycle_kernel
    numBlocks = (roots_count + blockSize - 1) / blockSize;
    break_cycle_kernel<<<numBlocks, blockSize>>>(d_last, succ, d_roots, roots_count, E);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize break_cycle_kernel");

    //apply list ranking on successor to get Euler tour
    CudaSimpleListRank(succ, devRank, E, euler.notAllDone, euler.devNotAllDone, euler.devRankNext);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    // std::cout << "emc phase 2: " << duration << " ms.\n";

    #ifdef DEBUG
        std::cout << "d_first array:" << std::endl;
        print_device_array(d_first, N);

        std::cout << "d_last array:" << std::endl;
        print_device_array(d_last, N);

        std::cout << "d_next array:" << std::endl;
        print_device_array(d_next, E);

        std::cout << "successor array:" << std::endl;
        print_device_array(succ, E);

        std::cout << "euler Path array:" << std::endl;
        print_device_array(devRank, E);
    #endif

    int *d_parent = euler.d_parent;

    numBlocks = (N + blockSize - 1) / blockSize;

    numBlocks = (E + blockSize - 1) / blockSize;
    find_parent<<<numBlocks, blockSize>>>(E, devRank, d_edges_to, d_edges_from, d_parent);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize find_parent kernel");

    // g_verbose = true;

    if(g_verbose) {
        std::cout << "Parent array from EulerianTour:" << std::endl;
        int* h_parent = new int[N];
        CUDA_CHECK(cudaMemcpy(h_parent, d_parent, sizeof(int) * N, cudaMemcpyDeviceToHost), 
            "Failed to copy back parent array");

        for(int i = 0; i < N; ++i) {
            std::cout << "parent["<< i << "]= " << h_parent[i] << "\n";
        }
        std::cout << std::endl;

        delete[] h_parent;
    }
}
