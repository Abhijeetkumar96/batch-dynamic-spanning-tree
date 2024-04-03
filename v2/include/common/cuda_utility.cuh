/******************************************************************************
* Functionality: GPU related Utility manager
 ******************************************************************************/

#ifndef CUDA_UTILITY_H
#define CUDA_UTILITY_H

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
extern bool checker;
extern bool g_verbose;
extern long maxThreadsPerBlock;

inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// #define DEBUG

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

// Function to print available and total memory
inline void printMemoryInfo(const std::string& message) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << message << ": Used GPU memory: " 
        << used_db / (1024.0 * 1024.0) << " MB, Free GPU Memory: " 
        << free_db / (1024.0 * 1024.0) << " MB, Total GPU Memory: " 
        << total_db / (1024.0 * 1024.0) << " MB" << std::endl;
}

inline void cuda_init(int device) {
    // Set the CUDA device
    CUDA_CHECK(cudaSetDevice(device), "Unable to set device ");
    cudaGetDevice(&device); // Get current device

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    cudaFree(0);

    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
}

template <typename T>
__global__ 
void print_device_array_kernel(T* array, long size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { // Let only a single thread do the printing
        for (int i = 0; i < size; ++i) {
            printf("Array[%d] = %d\n", i, array[i]);
        }
    }
}

template <typename T>
__global__ 
void print_device_edge_list_kernel(T* edgeList, long numEdges) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { 
        for (int i = 0; i < numEdges; ++i) {
            int u = edgeList[i] >> 32;  // Extract higher 32 bits
            int v = edgeList[i] & 0xFFFFFFFF; // Extract lower 32 bits
            printf("Edge[%d] = (%d, %d)\n", i, u, v);
        }
    }
}



void pointer_jumping(int* d_next, int n);

/*
    Scenario 1:
    - num_items             <-- [8]
    - d_in                  <-- [0, 2, 2, 9, 5, 5, 5, 8]
    - d_out                 <-- [0, 2, 9, 5, 8]
    - d_num_selected_out    <-- [5]    

    Scenario 2:
    - d_in                  <-- [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    - d_out                 <-- [0, 1, 0] 

    So we need to have the array sorted
*/
// void find_unique(
//     int* d_in, 
//     int* d_out,
//     int num_items,
//     int& h_num_selected_out);

template <typename T>
inline void host_print(const std::vector<T> arr) {
    for(const auto &i : arr)
        std::cout << i <<" ";
    std::cout << std::endl;
}

template <typename T>
inline void print_device_array(const T* arr, long size) {
    print_device_array_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_array_kernel");
}

template <typename T>
inline void print_device_edge_list(const T* arr, long size) {
    print_device_edge_list_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_edge_list_kernel");
}

template <typename T1, typename T2>
inline void find_unique(
    T1* d_in, 
    T1* d_out,
    T2 num_items,
    T2& h_num_selected_out) {
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T2* d_num_selected_out = nullptr;

    // Allocate device memory for storing the number of unique elements selected
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(T2)), "Failed to allocate memory for d_num_selected_out");

    // Query temporary storage requirements for sorting and selecting unique keys
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_in, d_in, num_items);
    size_t max_temp_storage_bytes = temp_storage_bytes;
    cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);
    max_temp_storage_bytes = std::max(max_temp_storage_bytes, temp_storage_bytes);

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, max_temp_storage_bytes), "Failed to allocate memory for d_num_selected_out");

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, max_temp_storage_bytes, d_in, d_in, num_items);

    // Run unique selection operation
    cub::DeviceSelect::Unique(d_temp_storage, max_temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);

    // Copy the number of unique elements selected back to host
    CUDA_CHECK(cudaMemcpy(&h_num_selected_out, d_num_selected_out, sizeof(T2), cudaMemcpyDeviceToHost), "Failed to allocate memory for d_num_selected_out");

    #ifdef DEBUG
        std::vector<T1> h_data(num_items);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_in, num_items * sizeof(T1), cudaMemcpyDeviceToHost), 
                   "Failed to copy sorted data back to host");

        std::cout << "Sorted Data:\n";
        for(const auto& val : h_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    #endif

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free d_temp_storage");
    CUDA_CHECK(cudaFree(d_num_selected_out), "Failed to free d_temp_storage");
}


#endif // CUDA_UTILITY_H