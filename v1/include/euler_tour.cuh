#ifndef EULER_TOUR_CUH
#define EULER_TOUR_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "cuda_utility.cuh"
#include "list_ranking.cuh"

class EulerianTour {
    public:
        int N, edges, edge_count;
        int *d_parent;
        int *d_child_count;
        int *d_child_num;
        int *starting_index;
        int *successor;
        int *d_child_list;
        int* d_last_edge;
        int* new_first;
        int* new_last;
        int* d_euler_tour_arr;
        int2 *d_edge_num;

        ListRanking* getListRanking() const {
            return listRanking;
        }

        EulerianTour(int nodes) : N(nodes) {
            
            edge_count = N - 1;
            edges = edge_count * 2;
            
            allocateMemory();
            initializeMemory();
        }

        ~EulerianTour() {
            freeMemory();
        }

    private:
        ListRanking* listRanking;

        void allocateMemory() {

            CUDA_CHECK(cudaMalloc(&d_parent, N * sizeof(int)),                          "Failed to allocate memory for d_parent");
            CUDA_CHECK(cudaMalloc(&d_child_count, N * sizeof(int)),                     "Failed to allocate memory for d_child_count");
            CUDA_CHECK(cudaMalloc(&d_child_num, N * sizeof(int)),                       "Failed to allocate memory for d_child_num");
            CUDA_CHECK(cudaMalloc(&starting_index, (N+1) * sizeof(int)),                "Failed to allocate memory for starting_index");
            CUDA_CHECK(cudaMalloc(&d_edge_num, edges * sizeof(int2)),                   "Failed to allocate memory for d_edge_num");
            CUDA_CHECK(cudaMalloc(&successor, edges * sizeof(int)),                     "Failed to allocate memory for successor");
            CUDA_CHECK(cudaMalloc(&d_child_list, edge_count * sizeof(int)),             "Failed to allocate memory for d_child_list");
            CUDA_CHECK(cudaMalloc(&d_euler_tour_arr, edges * sizeof(int)),              "Failed to allocate memory for d_euler_tour_arr");
            CUDA_CHECK(cudaMalloc(&new_first, N * sizeof(int)),                         "Failed to allocate memory for new_first");
            CUDA_CHECK(cudaMalloc(&new_last, N * sizeof(int)),                          "Failed to allocate memory for new_last");

            CUDA_CHECK(cudaMallocManaged(&d_last_edge, sizeof(int)),                    "Failed to allocate memory for d_last_edge");

            // Initialize ListRanking
            listRanking = new ListRanking(edges);
        }

        void initializeMemory() {
            CUDA_CHECK(cudaMemset(d_child_count, 0, N * sizeof(int)),                   "cudaMemset of d_child_count");
            CUDA_CHECK(cudaMemset(d_child_num, 0, N * sizeof(int)),                     "cudaMemset of d_child_num");
            CUDA_CHECK(cudaMemset(starting_index, 0, (N+1) * sizeof(int)),              "cudaMemset of starting_index");
            CUDA_CHECK(cudaMemset(d_child_list, 0, sizeof(int) * (N - 1)),              "cudaMemset of d_child_list");
        }

        void freeMemory() {
            CUDA_CHECK(cudaFree(d_parent), "Failed to free d_parent");
            CUDA_CHECK(cudaFree(d_child_count), "Failed to free d_child_count");
            CUDA_CHECK(cudaFree(d_child_num), "Failed to free d_child_num");
            CUDA_CHECK(cudaFree(starting_index), "Failed to free starting_index");
            CUDA_CHECK(cudaFree(d_edge_num), "Failed to free d_edge_num");
            CUDA_CHECK(cudaFree(successor), "Failed to free successor");
            CUDA_CHECK(cudaFree(d_child_list), "Failed to free d_child_list");
            CUDA_CHECK(cudaFree(d_last_edge), "Failed to free d_last_edge");
            CUDA_CHECK(cudaFree(new_first), "Failed to free new_first");
            CUDA_CHECK(cudaFree(new_last), "Failed to free new_last");
        }
};

void cal_first_last(int root, int* d_parent, EulerianTour& eulerTour);

#endif // EULER_TOUR_CUH