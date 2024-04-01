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

void find_unique(
    int* d_in, 
    int* d_out,
    int num_items,
    int h_num_selected_out);

#endif // CUDA_UTILITY_H