#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>

#include "dynamic_spanning_tree/update_ds.cuh"
#include "common/cuda_utility.cuh"

using namespace cub;

bool g_verbose = true;  // Whether to display input/output to console
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
    unsigned char* d_flags)       // size <- numEdges
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < delete_batch_size) {

        uint64_t t = d_edges_to_delete[tid];
        uint32_t u = (uint32_t)(t >> 32);
        uint32_t v = (uint32_t)(t & 0xFFFFFFFF); 

        // delete tree edges
        if(u == d_parent[v]) {
            d_parent[v] = v;
            long pos = binary_search(d_edge_list, num_edges, t);
            if(pos != -1) {
                d_flags[pos] = 0;
            }
        }

        else if(v == d_parent[u]) {
            d_parent[u] = u;
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

template <typename T>
void DisplayResults(T* arr, int num_items) {
    for(int i = 0; i < num_items; ++i) {
        printf("%llu ", (unsigned long long)arr[i]);
    }
    printf("\n");
}

void DisplayDeviceUint64Array(uint64_t* d_arr, int num_items) {
    // Allocate host memory for the copy
    uint64_t* h_arr = new uint64_t[num_items];
    
    // Copy data from device to host
    cudaMemcpy(h_arr, d_arr, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);
    
    std::cout << "Device h_in Array: ";
    for(int i = 0; i < num_items; ++i) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup host memory
    delete[] h_arr;
}

void DisplayDeviceUCharArray(unsigned char* d_arr, int num_items) {
    // Allocate host memory for the copy
    unsigned char* h_arr = new unsigned char[num_items];
    
    // Copy data from device to host
    cudaMemcpy(h_arr, d_arr, sizeof(unsigned char) * num_items, cudaMemcpyDeviceToHost);
    
    std::cout << "Device h_flags Array: ";
    for(int i = 0; i < num_items; ++i) {
        std::cout << static_cast<int>(h_arr[i]) << " "; // Cast to int for clearer output
    }
    std::cout << std::endl;
    
    // Cleanup host memory
    delete[] h_arr;
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

    #ifdef DEBUG
        DisplayDeviceUint64Array(d_in, num_items);
        DisplayDeviceUCharArray(d_flags, num_items);
    #endif

    long     *d_num_selected_out   = NULL;
    g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(long));

    // Allocate temporary storage
    void        *d_temp_storage = NULL;
    size_t      temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    long h_num;
    cudaMemcpy(&h_num, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost);
    std::cout << "\nh_num: " <<  h_num << std::endl;
    num_items = h_num;
    // Copy output data back to host
    uint64_t* h_out = new uint64_t[num_items];
    cudaMemcpy(h_out, d_out, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);

    #ifdef DEBUG
        // Print output data
        printf("\nOutput Data (h_out):\n");
        DisplayResults(h_out, h_num); // Print only the selected elements
    #endif

}

void update_edgelist(
    int* d_parent, int num_vert, 
    uint64_t* d_edge_list, uint64_t* d_updated_ed_list, long& num_edges, 
    uint64_t* d_edges_to_delete, int delete_size) {

    // sort the input edges
    sort_array_uint64_t(d_edge_list, num_edges);
    
    // init d_flag with true values
    unsigned char   *d_flags = NULL;
    std::vector<unsigned char> h_flags(num_edges, 1);
    CUDA_CHECK(cudaMalloc((void**)&d_flags, sizeof(unsigned char) * num_edges), 
        "Failed to allocate memory for d_flags");

    CUDA_CHECK(cudaMemcpy(d_flags, h_flags.data(), sizeof(unsigned char) * num_edges, cudaMemcpyHostToDevice),
        "Failed to copy back d_flags");

    int numThreads = 1024;
    int numBlocks = (delete_size + numThreads - 1) / numThreads;

    // Launch kernel to mark batch edges for deletion in the actual edge_list
    mark_delete_edges_kernel<<<numThreads, numBlocks>>>(
        d_parent, 
        d_edge_list, 
        num_edges, 
        d_edges_to_delete, 
        delete_size, 
        d_flags
    );

    numBlocks = (num_vert + numThreads - 1) / numThreads;

    mark_tree_edges_kernel<<<numThreads, numBlocks>>>(d_parent, d_flags, d_edge_list, num_edges, num_vert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to delete edges");

    // now delete the edges from the graph array
    select_flagged(d_edge_list, d_updated_ed_list, d_flags, num_edges);

    #ifdef DEBUG
        std::cout << "printing updated edgelist:\n";
        std::cout << "numEdges after delete batch: " << num_edges << "\n";
        print_device_edge_list(d_updated_ed_list, num_edges);
    #endif
}

