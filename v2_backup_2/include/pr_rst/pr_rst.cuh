#ifndef PR_RST_H
#define PR_RST_H

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#include "common/cuda_utility.cuh"

class RST_Resource_manager {
public:
    int numVert             = 0;
    int *d_winner           = nullptr;;
    int *d_ptr              = nullptr;
    int *d_parent_ptr       = nullptr;
    int *d_new_parent_ptr   = nullptr;
    int *d_pr_arr           = nullptr;
    int *d_OnPath           = nullptr;
    int *d_new_OnPath       = nullptr;
    int *d_marked_parent    = nullptr;
    int *d_next             = nullptr;
    int *d_new_next         = nullptr;
    int *d_index_ptr        = nullptr;
    int *d_pr_size_ptr      = nullptr;

    RST_Resource_manager(int n) {
        numVert = n;
        
        // Log2 computation and size determination
        std::cout << "log2(n) = " << std::log2(numVert) << std::endl;
        int log_2_size = std::ceil(std::log2(numVert));
        long long pr_size = std::ceil(numVert * 1LL * log_2_size);
        
        std::cout << "pr_size = " << pr_size << std::endl;
        long long size = numVert * 1LL * sizeof(int); // For n vertices
        std::cout << "size: " << size << std::endl;

        // Memory allocation
        CUDA_CHECK(cudaMalloc((void **)&d_ptr, size),                       "Failed to allocate memory for d_ptr");
        CUDA_CHECK(cudaMalloc((void **)&d_parent_ptr, size),                "Failed to allocate memory for d_new_parent_ptr");
        CUDA_CHECK(cudaMalloc((void **)&d_new_parent_ptr, size),            "Failed to allocate memory for d_new_parent_ptr");
        CUDA_CHECK(cudaMalloc((void **)&d_pr_arr, sizeof(int) * pr_size),   "Failed to allocate memory for d_pr_arr");
        CUDA_CHECK(cudaMalloc((void **)&d_OnPath, size),                    "Failed to allocate memory for d_OnPath");
        CUDA_CHECK(cudaMalloc((void **)&d_new_OnPath, size),                "Failed to allocate memory for d_new_OnPath");
        CUDA_CHECK(cudaMalloc((void **)&d_marked_parent, size),             "Failed to allocate memory for d_marked_parent");
        CUDA_CHECK(cudaMalloc((void **)&d_next, size),                      "Failed to allocate memory for d_next");
        CUDA_CHECK(cudaMalloc((void **)&d_new_next, size),                  "Failed to allocate memory for d_new_next");
        CUDA_CHECK(cudaMalloc((void **)&d_index_ptr, size),                 "Failed to allocate memory for d_index_ptr");
        CUDA_CHECK(cudaMalloc((void **)&d_pr_size_ptr, size),               "Failed to allocate memory for d_pr_size_ptr");
    }

    ~RST_Resource_manager() {
        // Free allocated device memory
        CUDA_CHECK(cudaFree(d_ptr),             "Failed to free d_ptr");
        CUDA_CHECK(cudaFree(d_parent_ptr),      "Failed to free d_parent_ptr");
        CUDA_CHECK(cudaFree(d_new_parent_ptr),  "Failed to free d_new_parent_ptr");
        CUDA_CHECK(cudaFree(d_pr_arr),          "Failed to free d_pr_arr");
        CUDA_CHECK(cudaFree(d_OnPath),          "Failed to free d_OnPath");
        CUDA_CHECK(cudaFree(d_new_OnPath),      "Failed to free d_new_OnPath");
        CUDA_CHECK(cudaFree(d_marked_parent),   "Failed to free d_marked_parent");
        CUDA_CHECK(cudaFree(d_next),            "Failed to free d_next");
        CUDA_CHECK(cudaFree(d_new_next),        "Failed to free d_new_next");
        CUDA_CHECK(cudaFree(d_index_ptr),       "Failed to free d_index_ptr");
        CUDA_CHECK(cudaFree(d_pr_size_ptr),     "Failed to free d_pr_size_ptr");
    }
};

#endif // PR_RST_H

// Example usage
// int main() {
//     int n = 1024; // Example size
//     RST_Resource_manager pr_resources(n);
// }

