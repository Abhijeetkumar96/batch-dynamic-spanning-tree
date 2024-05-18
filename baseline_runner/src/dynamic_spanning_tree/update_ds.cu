#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>

#include "common/Timer.hpp"
#include "dynamic_spanning_tree/update_ds.cuh"
#include "common/cuda_utility.cuh"

// #define CHECKER

using namespace cub;

CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

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
void print_variable(long* u) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) {
        printf("value = %ld\n", u);
    }
}

__global__
void mark_delete_edges_kernel(
    int* d_parent,          // size <- numNodes
    uint64_t* d_edge_list,  // size <- numEdges
    long num_edges,         
    uint64_t* d_edges_to_delete, // size <- delete_batch_size
    int delete_batch_size, 
    unsigned char* d_flags,     // size <- numEdges
    int* d_unique_rep,
    int* d_deleted_count,
    int root)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid == 0)
        d_unique_rep[0] = root;
    
    if (tid < delete_batch_size) {

        uint64_t t = d_edges_to_delete[tid];

        uint32_t u = (uint32_t)(t >> 32);
        uint32_t v = (uint32_t)(t & 0xFFFFFFFF); 

        // delete tree edges
        if(u == d_parent[v]) {
            // update the unique_rep here only
            #ifdef DEBUG
                printf("u: %d, v: %d\n", u, v);
            #endif
            int old_count = atomicAdd(d_deleted_count, 1);
            d_parent[v] = v;
            d_unique_rep[old_count + 1] = v;

            long pos = binary_search(d_edge_list, num_edges, t);
            if(pos != -1) {
                d_flags[pos] = 0;
            }
        }

        else if(v == d_parent[u]) {
            // update the unique_rep here only
            #ifdef DEBUG
                printf("u: %d, v: %d\n", u, v);
            #endif
            int old_count = atomicAdd(d_deleted_count, 1);

            d_parent[u] = u;
            d_unique_rep[old_count + 1] = u;

            long pos = binary_search(d_edge_list, num_edges, t);
            if(pos != -1) {
                d_flags[pos] = 0;
            }
        }

        else {
            // t is the key, to be searched in the d_edge_list array
            long pos = binary_search(d_edge_list, num_edges, t);
            if(pos != -1) {
                d_flags[pos] = 0;
            }
        }
    }
}

__device__ 
void device_swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__
void mark_tree_edges_kernel(
    int* d_parent, 
    unsigned char* d_flags, 
    uint64_t* d_edge_list, 
    long num_edges, 
    int num_vert) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_vert) {
        int u = tid;
        int v = d_parent[tid];

        if(u != v) {
            if(u > v) {
                device_swap(u, v);
            }

            uint64_t t = ((uint64_t)(u) << 32 | (v));
            long pos = binary_search(d_edge_list, num_edges, t);

            if(pos != -1) {
                d_flags[pos] = 0;
            }
        }
    }
}

void sort_array_uint64_t(uint64_t* d_data, long num_items) {
    // Allocate temporary storage for sorting
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);
    cudaDeviceSynchronize();
}

void select_flagged(uint64_t* d_in, uint64_t* d_out, unsigned char* d_flags, long& num_items) {

    if(g_verbose) {
        DisplayDeviceUint64Array(d_in, d_flags, num_items);
        // DisplayDeviceUCharArray(d_flags, num_items);
    }
    
    long *d_num_selected_out   = NULL;
    g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(long));

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    long h_num;
    cudaMemcpy(&h_num, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost);
    // std::cout << "\nh_num: " <<  h_num << std::endl;
    num_items = h_num;
    
    // if(g_verbose) {
    //     // Copy output data back to host
    //     uint64_t* h_out = new uint64_t[num_items];
    //     cudaMemcpy(h_out, d_out, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);

    //     // Print output data
    //     printf("\nOutput Data (h_out):\n");
    //     DisplayResults(h_out, h_num); // Print only the selected elements
    // }

}

void update_edgelist(
    int* d_parent,                  // -- 1
    int num_vert,                   // -- 2
    uint64_t* d_edge_list,          // -- 3
    uint64_t* d_updated_ed_list,    // -- 4
    long& num_edges,                // -- 5
    uint64_t* d_edges_to_delete,    // -- 6
    int delete_size,                // -- 7
    int* d_unique_rep,              // -- 8
    int& unique_rep_count,          // -- 9
    int root) {                     // -- 10

    std::cout << "numVert: " << num_vert << ", num_edges: " << num_edges * 2 << " and delete batch size: " << delete_size << std::endl;

    #ifdef CHECKER
        uint64_t* host_edge_list = new uint64_t[num_edges];
        CUDA_CHECK(cudaMemcpy(host_edge_list, d_edge_list, num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "Failed to copy back edges");
    #endif

    // init d_flag with true values
    unsigned char   *d_flags = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_flags, sizeof(unsigned char) * num_edges), 
        "Failed to allocate memory for d_flags");

    CUDA_CHECK(cudaMemset(d_flags, 1, sizeof(unsigned char) * num_edges), "Failed to set d_flags");

    int* d_deleted_count;
    cudaMalloc((void**)&d_deleted_count, sizeof(int));
    cudaMemset(d_deleted_count, 0, sizeof(int));

    // sort the input edges (assumed in pre-processing time)
    sort_array_uint64_t(d_edge_list, num_edges);

    // std::cout << "Sorting edge list over.\n";

    #ifdef CHECKER
        std::cout << "Checker is active in update_ds.cu, in update_edgelist func.\n";
        uint64_t* h_edge_list = new uint64_t[num_edges];
        CUDA_CHECK(cudaMemcpy(h_edge_list, d_edge_list, num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "Failed to copy back edges");

        std::cout << "Is sorted from update_edgelist: " << std::is_sorted(h_edge_list, h_edge_list + num_edges) << std::endl;

        std::sort(host_edge_list, host_edge_list + num_edges);

        bool no_err = false;
        for(long i = 0; i < num_edges; ++i) {
            if(host_edge_list[i] != h_edge_list[i]) {
                no_err = true;
                std::cerr << "Error in sorted array from update_edgelist_bfs.\n";
            }
        }

        if(!no_err) {
            std::cout << "No errors! Array sorted correctly.\n";
            for(long i = 0; i < 10; ++i) {
                std::cout << host_edge_list[i] << " ";
            }

            std::cout << std::endl;
            for(long i = 0; i < 10; ++i) {
                std::cout << h_edge_list[i] << " ";
            }
        }

        std::cout << "\nChecking over, all good.\n";

        delete[] h_edge_list;
        delete[] host_edge_list;
    #endif

    int numThreads = 1024;
    int numBlocks = (delete_size + numThreads - 1) / numThreads;

    // start timer here
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel to mark batch edges for deletion in the actual edge_list
    mark_delete_edges_kernel<<<numBlocks, numThreads>>>(
        d_parent, 
        d_edge_list, 
        num_edges, 
        d_edges_to_delete, 
        delete_size, 
        d_flags,
        d_unique_rep,
        d_deleted_count,
        root
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after mark_delete_edges_kernel");

    numBlocks = (num_vert + numThreads - 1) / numThreads;

    mark_tree_edges_kernel<<<numThreads, numBlocks>>>(d_parent, d_flags, d_edge_list, num_edges, num_vert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after mark_tree_edges_kernel");

    CUDA_CHECK(cudaMemcpy(&unique_rep_count, d_deleted_count, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back unique_rep_count");

    // now delete the edges from the graph array
    select_flagged(d_edge_list, d_updated_ed_list, d_flags, num_edges);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("update data structure", duration);
    // std::cout << "Time taken to update the datastr: " << duration << " ms.\n";

    // std::cout << "Number of deleted tree edges from update_ds: " << unique_rep_count << std::endl;
    unique_rep_count++;
    // std::cout << "Number of unique_rep_count from update_ds: " << unique_rep_count << std::endl;

    // std::cout << "printing updated edgelist:\n";
    // std::cout << "numEdges after deleting tree edges and batch B of edges: " << num_edges << "\n";

    if(g_verbose) {
        print_device_edge_list(d_updated_ed_list, num_edges);
    }
    // Clean up
    CUDA_CHECK(cudaFree(d_deleted_count), "Failed to free d_deleted_count");
    CUDA_CHECK(cudaFree(d_flags), "Failed to free d_flags");
}
