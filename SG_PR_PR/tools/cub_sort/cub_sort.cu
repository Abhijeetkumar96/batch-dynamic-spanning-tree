#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm> // For std::is_sorted

// Utility function to check CUDA errors
inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

template <typename T1, typename T2>
inline void cub_sorting(T1* d_in, T2 num_items) {
    
    size_t temp_storage_bytes   =   0;
    void* d_temp_storage        =   nullptr;
    T2* d_num_selected_out      =   nullptr;

    // Allocate device memory for storing the number of unique elements selected
    checkCuda(cudaMalloc((void**)&d_num_selected_out, sizeof(T2)));
    
    // Query temporary storage requirements for sorting
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
    // Allocate temporary storage
    checkCuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

    checkCuda(cudaFree(d_temp_storage));

    bool print = true;
    
    // print d_in values 
    if(print) {
        int* h_in = new int[num_items];
        checkCuda(cudaMemcpy(h_in, d_in, num_items * sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "sorted array output:\n";
        for(int i = 0; i < num_items; ++i) {
            std::cout << h_in[i] << "\n";
        }
        std::cout << std::endl;

        if (std::is_sorted(h_in, h_in + num_items)) {
            std::cout << "Sorted in the range." << std::endl;
        } else {
            std::cout << "Not Sorted in the range. " << std::endl;
        }
    }

    cudaFree(d_temp_storage);
}

int main() {

    int num_items = 100;
    std::vector<int> h_data(num_items);

    // Random number generation
    std::mt19937_64 rng; // A Mersenne Twister pseudo-random generator of 64-bit numbers
    std::uniform_int_distribution<int> dist(0, num_items); // Distribution range

    // Generate random numbers
    for (size_t i = 0; i < num_items; ++i) {
        h_data[i] = dist(rng);
    }

    // Allocate and copy data from host to device
    int *d_data;
    cudaMalloc(&d_data, num_items * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);

    cub_sorting(d_data, num_items);

    cudaMemcpy(h_data.data(), d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    checkCuda(cudaGetLastError());

    if (!std::is_sorted(h_data.begin(), h_data.end())) {
        std::cerr << "Data not sorted correctly." << std::endl;
    } else {
        std::cout << "Data sorted correctly." << std::endl;
    }

    // Clean up
    cudaFree(d_data);
}