#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

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

void print(const std::vector<int> &arr);
void print(const thrust::device_vector<int> &arr);
void printArr(const std::vector<int> &parent,int vertices,int stride);
void printPR(const std::vector<int> &pr_arr,const std::vector<int> &pr_arr_size,int vertices,int log_2_size);

template <typename T>
inline void print_device_array(const T* arr, long size) {
    print_device_array_kernel<<<1, 1>>>(arr, size);
    cudaDeviceSynchronize();
}