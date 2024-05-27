#ifndef EULER_H
#define EULER_H

#include <iostream>
#include "common/cuda_utility.cuh"

namespace mce {

class EulerianTour {
public:
    int2 *d_edge_num;
    int *d_child_count;
    int *d_child_num;
    int *starting_index;
    int *d_successor;
    int *d_euler_tour_arr;
    int *d_child_list;
    int *d_first_edge;
    int *d_rank;
    int *d_new_first;
    int *d_new_last;
    
    int N;
    int edges;
    int edge_count;
    int num_comp;
    
    // list_ranking params
    unsigned long long *devRankNext;
    int *devNotAllDone;
    int *notAllDone;

    // Constructor declaration
    EulerianTour(int nodes, int numComponents) : N(nodes), num_comp(numComponents) {
        edge_count = N - num_comp; 
        edges = edge_count * 2;
        
        mem_alloc();
        mem_init();
    }
    // Destructor declaration
    ~EulerianTour() {
        // Free allocated device memory
        cudaFree(d_child_count);
        cudaFree(d_child_num);
        cudaFree(starting_index);
        cudaFree(d_edge_num);
        cudaFree(d_successor);
        cudaFree(d_child_list);
        cudaFree(d_first_edge);
        cudaFree(d_rank);
        cudaFree(d_new_first);
        cudaFree(d_new_last);
    }
    
    void mem_alloc() {
        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_child_count, N * sizeof(int)), "Failed to allocate device memory for d_child_count");
        CUDA_CHECK(cudaMalloc(&d_child_num, N * sizeof(int)), "Failed to allocate device memory for d_child_num");
        CUDA_CHECK(cudaMalloc(&starting_index, (N+1) * sizeof(int)), "Failed to allocate memory for starting_index");
        CUDA_CHECK(cudaMalloc(&d_edge_num,  edges * sizeof(int2)), "Failed to allocate memory for d_edge_num");
        CUDA_CHECK(cudaMalloc(&d_successor, edges * sizeof(int)), "Failed to allocate memory for d_successor");
        CUDA_CHECK(cudaMalloc(&d_euler_tour_arr, edges * sizeof(int)), "Failed to allocate memory for d_successor");
        CUDA_CHECK(cudaMalloc(&d_child_list, edge_count * sizeof(int)), "Failed to allocate memory for d_child_list");
        CUDA_CHECK(cudaMalloc(&d_first_edge, num_comp * sizeof(int)), "Failed to allocate memory for d_first_edge");
        CUDA_CHECK(cudaMalloc(&d_rank, edges * sizeof(int)), "Failed to allocate device memory for d_rank");
        CUDA_CHECK(cudaMalloc(&d_new_first, N * sizeof(int)), "Failed to allocate device memory for d_new_first");
        CUDA_CHECK(cudaMalloc(&d_new_last, N * sizeof(int)), "Failed to allocate device memory for d_new_last");

        // list_ranking params
        CUDA_CHECK(cudaMalloc((void **)&devRankNext, sizeof(unsigned long long) * edges), "Failed to allocate devRankNext");
        CUDA_CHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)), "Failed to allocate devNotAllDone");
        CUDA_CHECK(cudaMallocHost(&notAllDone, sizeof(int)), "Failed to allocate notAllDone");
    }
    void mem_init() {
        // Initialize device memory
        CUDA_CHECK(cudaMemset(d_child_list, 0, edge_count * sizeof(int)), "Failed to initialize d_child_list");
        CUDA_CHECK(cudaMemset(d_child_count, 0, N * sizeof(int)), "Failed to initialize d_child_count");
        CUDA_CHECK(cudaMemset(d_child_num, 0, N * sizeof(int)), "Failed to initialize d_child_num");
        CUDA_CHECK(cudaMemset(starting_index, 0, (N+1) * sizeof(int)), "Failed to initialize starting_index");
    }
};

void cal_first_last(
    int* d_parent, 
    int* d_roots, 
    int* d_rep_arr, 
    int* d_rep_map, 
    int nodes, 
    int num_comp, 
    EulerianTour* euler_mag);

}// namespace mce

#endif // EULER_H
