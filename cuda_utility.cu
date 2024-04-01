#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

#include "cuda_utility.h"

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

void pointer_jumping(int* d_next, int n)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
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

void find_unique(
    int* d_in, 
    int* d_out,
    int num_items,
    int h_num_selected_out) {
    
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

    #ifdef DEBUG
        std::vector<int> h_data(num_items);
        // Copy sorted data back to host
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_in, num_items * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back data");
        
        std::cout <<"h_data:\n";
        for(auto i : h_data)
            std::cout << i << " ";
    #endif

    int* d_num_selected_out;
    cudaMalloc((void**)&d_num_selected_out, sizeof(int));

    temp_storage_bytes = 0;
    
    // Query temporary storage requirements for unique selection
    cub::DeviceSelect::Unique(NULL, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);

    // Run unique selection operation
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);

    // h_num_selected_out <- Host variable to store the number of unique elements selected
    CUDA_CHECK(cudaMemcpy(&h_num_selected_out, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost), 
        "Error in copying d_num_selected_out");

    int* h_out = new int[h_num_selected_out]; // Allocate host memory for the unique elements
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * h_num_selected_out, cudaMemcpyDeviceToHost),
        "Error in copying h_out");

    std::cout << "\nUnique elements (" << h_num_selected_out << "): ";
    for (int i = 0; i < h_num_selected_out; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
}
