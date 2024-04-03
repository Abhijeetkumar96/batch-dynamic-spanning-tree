#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

#include "common/cuda_utility.cuh"

#define DEBUG

__global__
void pointer_jumping_kernel(int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n) {

        if(next[tid] != tid){
            next[tid] = next[next[tid]];
        }
    }
}

void pointer_jumping(int* d_next, int n) {

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto parallel_start = std::chrono::high_resolution_clock::now();  
    for (int j = 0; j < std::ceil(std::log2(n)); ++j)
    {
        pointer_jumping_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, n);
        cudaDeviceSynchronize();
    }

    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(parallel_end - parallel_start).count();
    printf("Total time for parallel pointer jumping : %ld microseconds (%d number of keys)\n", parallel_duration, n);  
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

