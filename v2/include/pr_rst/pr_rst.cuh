#ifndef PR_RST_H
#define PR_RST_H

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

class RST_Resource_manager {
public:
    int numVert             = 0;
    int *d_winner           = nullptr;;
    int *d_ptr              = nullptr;
    int *d_parent_ptr       = nullptr;
    int *d_new_parent_ptr   = nullptr;
    int *d_pr_arr           = nullptr;
    int *d_label            = nullptr;
    int *d_OnPath           = nullptr;
    int *d_new_OnPath       = nullptr;
    int *d_rep              = nullptr;
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
        cudaMalloc((void **)&d_ptr, size);
        cudaMalloc((void **)&d_parent_ptr, size);
        cudaMalloc((void **)&d_new_parent_ptr, size);
        cudaMalloc((void **)&d_pr_arr, sizeof(int) * pr_size);
        cudaMalloc((void **)&d_label, size);
        cudaMalloc((void **)&d_rep, size);
        cudaMalloc((void **)&d_OnPath, size);
        cudaMalloc((void **)&d_new_OnPath, size);
        cudaMalloc((void **)&d_marked_parent, size);
        cudaMalloc((void **)&d_next, size);
        cudaMalloc((void **)&d_new_next, size);
        cudaMalloc((void **)&d_index_ptr, size);
        cudaMalloc((void **)&d_pr_size_ptr, size);
    }

    ~RST_Resource_manager() {
        // Free allocated device memory
        cudaFree(d_ptr);
        cudaFree(d_parent_ptr);
        cudaFree(d_new_parent_ptr);
        cudaFree(d_pr_arr);
        cudaFree(d_label);
        cudaFree(d_rep);
        cudaFree(d_OnPath);
        cudaFree(d_new_OnPath);
        cudaFree(d_marked_parent);
        cudaFree(d_next);
        cudaFree(d_new_next);
        cudaFree(d_index_ptr);
        cudaFree(d_pr_size_ptr);
    }
};

#endif // PR_RST_H

// Example usage
// int main() {
//     int n = 1024; // Example size
//     RST_Resource_manager pr_resources(n);
// }

