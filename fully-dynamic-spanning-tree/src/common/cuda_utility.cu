#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

#include "common/cuda_utility.cuh"

// #define DEBUG

using namespace cub;

CachingDeviceAllocator g_allocator_(true); // Caching allocator for device memory


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

__global__
void pointer_jumping_kernel(int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {

        if(next[tid] != tid)
        {
            next[tid] = next[next[tid]];
        }
    }
}

void pointer_jumping(int* d_next, int n) {

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // p_jump_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_next);
    // CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize p_jump");

    for (int j = 0; j < std::ceil(std::log2(n)); ++j){
        pointer_jumping_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, n);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize p_jump");
    }
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
    long num_items) {
    
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
    void* d_temp_storage = nullptr;
    
    cudaError_t status;
    status = cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_merged, d_merged, num_items);
    CUDA_CHECK(status, "Error in CUB SortKeys");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temp storage");
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
    int*&           d_values_out)         // Output values (processed edges' second vertices)

{
    // std::cout <<"num_items before edge_cleanup: " << num_items << "\n";
    cudaError_t status;
    
    // if(g_verbose) {
    //     std::cout << "printing from remove_self_loops_duplicates:\n";
    //     DisplayDeviceEdgeList(d_keys, d_values, num_items);
    // }

    radix_sort_for_pairs(d_keys, d_values, d_merged, num_items);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    
    // Mark self-loops and duplicates for removal
    long numThreads = 1024;
    long numBlocks = (num_items + numThreads - 1) / numThreads;
    markForRemoval<<<numBlocks, numThreads>>>(d_keys, d_values, d_flags, num_items);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    size_t temp_storage_bytes = 0;

    // Remove marked edges
    // Determine temporary storage requirements for selection
    void* d_temp_storage = NULL;
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate memory");
    // One call for keys
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    
    d_temp_storage = NULL;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate memory");
    // One call for values
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_values, d_flags, d_values_out, d_num_selected_out, num_items);
    CUDA_CHECK(status, "Error in CUB Flagged");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    // g_verbose = false;
    
    if(g_verbose) {
        std::cout <<"NumEdges after cleaning up: " << *d_num_selected_out << "\n";
        std::cout <<"Cleaned edge stream:\n";
        DisplayDeviceEdgeList(d_keys_out, d_values_out, *d_num_selected_out);
    }
}

// Kernel to find roots and count components
__global__ 
void find_root(const int* d_parent, int* d_root, int* d_num_comp, int num_vert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vert) {
        if (d_parent[idx] == idx) {
            atomicAdd(d_num_comp, 1);
            if (*d_root == -1) {
                *d_root = idx;
            }
        }
    }
}

bool is_tree_or_forest(const int* d_parent, const int num_vert, int& root) {
    int* d_num_comp = nullptr;
    int* d_root = nullptr;

    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&d_num_comp, sizeof(int)), "Failed to allocate d_num_comp");
    CUDA_CHECK(cudaMallocManaged(&d_root, sizeof(int)), "Failed to allocate d_root");

    // Initialize memory
    *d_num_comp = 0;
    *d_root = -1;

    int block_size = 1024;
    int num_blocks = (num_vert + block_size - 1) / block_size;

    // Launch the kernel
    find_root<<<num_blocks, block_size>>>(d_parent, d_root, d_num_comp, num_vert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize find_root kernel");

    // Determine if the structure is a tree
    bool is_tree = (*d_num_comp == 1);
    if (is_tree) {
        root = *d_root;
    } else {
        root = -1;  // Indicate that there is no single root
    }

    // Free allocated memory
    cudaFree(d_num_comp);
    cudaFree(d_root);

    return is_tree;
}

void select_flagged(int* d_in, int* d_out, unsigned char* d_flags, int& num_items) {

    int *d_num_selected_out   = NULL;
    g_allocator_.DeviceAllocate((void**)&d_num_selected_out, sizeof(int));

    g_verbose = false;

    if(g_verbose) {
        DisplayDeviceintArray(d_in, d_flags, num_items);
        DisplayDeviceUCharArray(d_flags, num_items);
    }

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator_.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    int h_num;
    CUDA_CHECK(cudaMemcpy(&h_num, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy back d_num_selected_out");
    // std::cout << "\nh_num: " <<  h_num << std::endl;
    num_items = h_num;
}

void select_flagged(uint64_t* d_in, uint64_t* d_out, unsigned char* d_flags, long& num_items) {

    if(g_verbose) {
        DisplayDeviceUint64Array(d_in, d_flags, num_items);
        DisplayDeviceUCharArray(d_flags, num_items);
    }
    
    long *d_num_selected_out   = NULL;
    g_allocator_.DeviceAllocate((void**)&d_num_selected_out, sizeof(long));

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator_.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    long h_num;
    CUDA_CHECK(cudaMemcpy(&h_num, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost),
        "Failed to copy back d_num_selected_out");
    // std::cout << "\nh_num: " <<  h_num << std::endl;
    num_items = h_num;
    
    if(g_verbose) {
        // Copy output data back to host
        uint64_t* h_out = new uint64_t[num_items];
        cudaMemcpy(h_out, d_out, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);

        // Print output data
        printf("\nOutput Data (h_out):\n");
        DisplayResults(h_out, h_num); // Print only the selected elements
    }

}

