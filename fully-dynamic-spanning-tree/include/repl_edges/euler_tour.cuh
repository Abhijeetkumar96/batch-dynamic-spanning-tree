#ifndef EULER_TOUR_CUH
#define EULER_TOUR_CUH

#include <cuda_runtime.h>
#include <iostream>

#include <thrust/sequence.h>

#include "common/cuda_utility.cuh"

class Euler_Tour {
public:

    int num_edges;
    int roots_count;
    int num_vertices;

    // Device pointers
    uint64_t *d_sorted_keys = nullptr;
    uint64_t *d_merged = nullptr;
    uint64_t *d_merged_keys_sorted = nullptr;
    uint64_t *d_indices = nullptr;
    int *d_u = nullptr;
    int *d_v = nullptr;
    int *d_edges_to = nullptr;
    int *d_edges_from = nullptr;
    uint64_t *d_index = nullptr;
    int *d_next = nullptr;

    // dont free memory for d_parent
    int *d_parent = nullptr;
    int *d_first = nullptr;
    int *d_last = nullptr;
    int *succ = nullptr;
    int *devRank = nullptr;

    // list_ranking params
    unsigned long long *devRankNext = nullptr;
    int *devNotAllDone = nullptr;
    int *notAllDone = nullptr;

    // Constructor
    Euler_Tour(int numVertices, int numEdges, int rootsCount) :
        num_edges(numEdges), roots_count(rootsCount), num_vertices(numVertices) {
        allocateMemory();
        initializeMemory();
    }

    // Destructor
    ~Euler_Tour() {
        freeMemory();
    }

    void allocateMemory() {
        int E = num_edges * 2;

        // Allocate memory for each array
        CUDA_CHECK(cudaMalloc(&d_sorted_keys, sizeof(uint64_t) * num_edges), "Failed to allocate memory for sorted edgelist");
        
        CUDA_CHECK(cudaMalloc(&d_u, sizeof(int) * num_edges),"Failed to allocate d_u");
        CUDA_CHECK(cudaMalloc(&d_v, sizeof(int) * num_edges), "Failed to allocate d_v");
        
        CUDA_CHECK(cudaMalloc(&d_edges_to,   sizeof(int) * E), "Failed to allocate d_edges_to");
        CUDA_CHECK(cudaMalloc(&d_edges_from, sizeof(int) * E),"Failed to allocate d_edges_from");

        CUDA_CHECK(cudaMalloc(&d_index,     sizeof(uint64_t) * E), "Failed to allocate d_index");
        CUDA_CHECK(cudaMalloc(&d_next,      sizeof(int) * E), "Failed to allocate d_next");

        CUDA_CHECK(cudaMalloc(&d_first,     sizeof(int) * num_vertices), "Failed to allocate d_first");
        CUDA_CHECK(cudaMalloc(&d_last,      sizeof(int) * num_vertices), "Failed to allocate d_last");

        CUDA_CHECK(cudaMalloc(&d_parent,    sizeof(int) * num_vertices), "Failed to allocate d_parent");

        CUDA_CHECK(cudaMalloc(&succ,        sizeof(int) * E), "Failed to allocate succ");
        CUDA_CHECK(cudaMalloc(&devRank,     sizeof(int) * E),"Failed to allocate devRank");

        CUDA_CHECK(cudaMalloc(&d_merged, sizeof(uint64_t) * E), "Failed to allocate d_merged");
        CUDA_CHECK(cudaMalloc(&d_merged_keys_sorted, sizeof(uint64_t) * E), "Failed to allocate d_merged");
        CUDA_CHECK(cudaMalloc(&d_indices, sizeof(uint64_t) * E), "Failed to allocate d_indices");

        // list_ranking params
        CUDA_CHECK(cudaMalloc((void **)&devRankNext, sizeof(unsigned long long) * E), "Failed to allocate devRankNext");
        CUDA_CHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)), "Failed to allocate devNotAllDone");
        CUDA_CHECK(cudaMallocHost(&notAllDone, sizeof(int)), "Failed to allocate notAllDone");
    }

    void initializeMemory() {
        // Initialize some arrays to specific values if necessary
        CUDA_CHECK(cudaMemset(d_first, -1, sizeof(int) * num_vertices), "Failed to memset d_first");
        CUDA_CHECK(cudaMemset(d_last, -1, sizeof(int) * num_vertices), "Failed to memset d_last");
        
        thrust::sequence(thrust::device, d_parent, d_parent + num_vertices);
    }

    void freeMemory() {
        // Free each allocated array
        CUDA_CHECK(cudaFree(d_sorted_keys), "Failed to free d_sorted_keys");
        CUDA_CHECK(cudaFree(d_u),           "Failed to free d_u");
        CUDA_CHECK(cudaFree(d_v),           "Failed to free d_v");
        CUDA_CHECK(cudaFree(d_edges_to),    "Failed to free d_edges_to");
        CUDA_CHECK(cudaFree(d_edges_from),  "Failed to free d_edges_from");
        CUDA_CHECK(cudaFree(d_index),       "Failed to free d_index");
        CUDA_CHECK(cudaFree(d_next),        "Failed to free d_next");
        CUDA_CHECK(cudaFree(d_first),       "Failed to free d_first");
        CUDA_CHECK(cudaFree(d_last),        "Failed to free d_last");
        CUDA_CHECK(cudaFree(succ),          "Failed to free succ");
        CUDA_CHECK(cudaFree(devRank),       "Failed to free devRank");
    }
};

void cuda_euler_tour(uint64_t* d_edge_list, int N, int num_edges, int* d_roots, int roots_count, Euler_Tour& euler);

#endif // EULER_TOUR_CUH