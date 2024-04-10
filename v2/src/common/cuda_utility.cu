#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

#include "common/cuda_utility.cuh"

#define DEBUG

using namespace cub;

__global__ 
void p_jump_kernel(int n, int *d_next) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid < n) {
        if(d_next[tid] != tid) {
            while (d_next[tid] != d_next[d_next[tid]])
            {
                d_next[tid] = d_next[d_next[tid]];
            }
        }
    }
}

void pointer_jumping(int* d_next, int n) {

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto parallel_start = std::chrono::high_resolution_clock::now();  
    
    p_jump_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_next);
    cudaDeviceSynchronize();

    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(parallel_end - parallel_start).count();
    printf("Total time for parallel pointer jumping : %ld microseconds (%d number of keys)\n", parallel_duration, n);  
}

// CUDA kernel to merge two integer arrays into an array of int64_t
__global__ 
void packPairs(const int *arrayU, const int *arrayV, uint64_t *arrayE, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Cast to int64_t to ensure the shift operates on 64 bits
        uint64_t u = arrayU[idx];
        uint64_t v = arrayV[idx];

        // Manual swap if u is greater than v
        if (u > v) {
            uint64_t temp = u;
            u = v;
            v = temp;
        }

        // Ensure 'v' is treated as a 64-bit value
        arrayE[idx] = (u << 32) | (v & 0xFFFFFFFFLL);
    }
}

__global__ 
void unpackPairs(const uint64_t *zippedArray, int *arrayA, int *arrayB, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Extract the upper 32 bits
        arrayA[idx] = zippedArray[idx] >> 32;
        // Extract the lower 32 bits, ensuring it's treated as a signed int
        arrayB[idx] = int(zippedArray[idx] & 0xFFFFFFFFLL);  
    }
}

void radix_sort_for_pairs(
    int* d_keys, 
    int* d_values, 
    uint64_t *d_merged, 
    long num_items, 
    void* d_temp_storage) {
    
    long threadsPerBlock = 1024;
    long blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;

    if(g_verbose) {
        std::cout << "printing from radix_sort_for_pairs before sorting:\n";
        DisplayDeviceEdgeList(d_keys, d_values, num_items);
    }
    
    packPairs<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_values, d_merged, num_items);
    CUDA_CHECK(cudaGetLastError(), "packPairs kernel launch failed");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after packPairs");
    // Sort the packed pairs
    size_t temp_storage_bytes = 0;
    
    cudaError_t status;
    status = cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_merged, d_merged, num_items);
    CUDA_CHECK(status, "Error in CUB SortKeys");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    status = cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_merged, d_merged, num_items);
    CUDA_CHECK(status, "Error in CUB SortKeys");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    // Kernel invocation for unpacking
    unpackPairs<<<blocksPerGrid, threadsPerBlock>>>(d_merged, d_keys, d_values, num_items);
    CUDA_CHECK(cudaGetLastError(), "unpackPairs kernel launch failed");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    if(g_verbose) {
        std::cout <<"Displaying sorted edgeList before edge_cleanup\n";
        DisplayDeviceEdgeList(d_keys, d_values, num_items);
    }
}

/****************************** Sorting ends ****************************************/

// Kernel to mark self-loops and duplicates
__global__ 
void markForRemoval(int* edges_u, int* edges_v, unsigned char* flags, size_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        // Mark self-loops
        if (edges_u[idx] == edges_v[idx]) {
            flags[idx] = false;
        }
        // Mark duplicates 
        else if (idx > 0 && edges_u[idx] == edges_u[idx - 1] && edges_v[idx] == edges_v[idx - 1]) {
            flags[idx] = false;
        }
        else {
            flags[idx] = true;
        }
    }
}

// Function to remove self-loops and duplicates from graph edges
void remove_self_loops_duplicates(
    int*&           d_keys,               // Input keys (edges' first vertices)
    int*&           d_values,             // Input values (edges' second vertices)
    int             num_items,
    uint64_t*&      d_merged,             // Intermediate storage for merged (zipped) keys and values
    unsigned char*& d_flags,              // Flags used to mark items for removal
    int*            d_num_selected_out,   // Output: number of items selected (non-duplicates, non-self-loops)
    int*&           d_keys_out,           // Output keys (processed edges' first vertices)
    int*&           d_values_out,         // Output values (processed edges' second vertices)
    void*&          d_temp_storage)       // Temporary storage for intermediate computations

{
    std::cout <<"num_items before edge_cleanup: " << num_items << "\n";
    cudaError_t status;
    
    if(g_verbose) {
        std::cout << "printing from remove_self_loops_duplicates:\n";
        DisplayDeviceEdgeList(d_keys, d_values, num_items);
    }

    radix_sort_for_pairs(d_keys, d_values, d_merged, num_items, d_temp_storage);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    
    // Mark self-loops and duplicates for removal
    long numThreads = 1024;
    long numBlocks = (num_items + numThreads - 1) / numThreads;
    markForRemoval<<<numBlocks, numThreads>>>(d_keys, d_values, d_flags, num_items);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    size_t temp_storage_bytes = 0;

    // Remove marked edges
    // Determine temporary storage requirements for selection
    status = DeviceSelect::Flagged(NULL, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    // One call for keys
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    // One call for values
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_values, d_flags, d_values_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    std::cout <<"NumEdges after cleaning up: " << num_items << "\n";
    std::cout <<"Cleaned edge stream:\n";
    
    if(g_verbose)
        DisplayDeviceEdgeList(d_keys_out, d_values_out, *d_num_selected_out);
}


// old find unique code
// void find_unique(
//     int* d_in, 
//     int* d_out,
//     int num_items,
//     int& h_num_selected_out) {
    
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;

    
//     // Allocate device memory for storing the number of unique elements selected
//     int* d_num_selected_out;
//     CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(int)), "Failed to allocate d_num_selected_out");
    
//     // Query temporary storage requirements for sorting and selecting unique keys
//     cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_in, d_in, num_items);
//     size_t max_temp_storage_bytes = temp_storage_bytes;
//     cub::DeviceSelect::Unique(NULL, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);
//     max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

//     // Allocate temporary storage
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, max_temp_storage_bytes), "Failed to allocate temporary storage");

//     // Run sorting operation
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, max_temp_storage_bytes, d_in, d_in, num_items);

//     // Run unique selection operation
//     cub::DeviceSelect::Unique(d_temp_storage, max_temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);

//     // Copy the number of unique elements selected back to host
//     CUDA_CHECK(cudaMemcpy(&h_num_selected_out, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost), 
//                "Failed to copy d_num_selected_out");

//     // Debugging: Print sorted data and unique elements if DEBUG is defined
//     #ifdef DEBUG
//         std::vector<int> h_data(num_items);
//         CUDA_CHECK(cudaMemcpy(h_data.data(), d_in, num_items * sizeof(int), cudaMemcpyDeviceToHost), 
//                    "Failed to copy sorted data back to host");

//         std::cout << "Sorted Data:\n";
//         for(auto val : h_data) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     #endif

//     // Cleanup
//     if (d_temp_storage) cudaFree(d_temp_storage);
//     if (d_num_selected_out) cudaFree(d_num_selected_out);
// }

