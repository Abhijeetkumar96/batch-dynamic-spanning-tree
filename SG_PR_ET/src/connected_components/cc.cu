#include <set>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

#include "connected_components/cc.cuh"
#include "common/Timer.hpp"

// #define DEBUG

using namespace cub;

CachingDeviceAllocator _g_allocator_(true);  // Caching allocator for device memory

__global__
void initialise(int* parent, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n) {
        parent[tid] = tid;
    }
}

__global__ 
void hooking(long numEdges, int* original_u, int* original_v, int* d_rep, int* d_flag, int itr_no) 
{
    long tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < numEdges) {
        
        int edge_u = original_u[tid];
        int edge_v = original_v[tid];

        int comp_u = d_rep[edge_u];
        int comp_v = d_rep[edge_v];

        if(comp_u != comp_v) 
        {
            *d_flag = 1;
            int max = (comp_u > comp_v) ? comp_u : comp_v;
            int min = (comp_u < comp_v) ? comp_u : comp_v;

            if(itr_no%2) {
                d_rep[min] = max;
            }
            else { 
                d_rep[max] = min;
            }
        }
    }
}

__global__ 
void short_cutting(int n, int* d_parent) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n) {
        if(d_parent[tid] != tid) {
            d_parent[tid] = d_parent[d_parent[tid]];
        }
    }   
}

__global__ 
void print_list(int* u, int* v, long numEdges) {
    
    long tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid == 0) {
        for(long i = 0; i < numEdges; ++i) {
            printf("edge[%ld]: %d, %d\n", i, u[i], v[i]);
        }
    }
}

std::string cc_get_filename(const std::string& path) {
    return std::filesystem::path(path).stem().string();
}

__global__
void select_rep(int* d_rep, unsigned char* d_flags, int numVert) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < numVert) {
        d_flags[d_rep[tid]] = 1;
    }

}

std::pair<int*, int> cc(int* edge_u, int* edge_v, int numVert, long numEdges) {

    const long numThreads = 1024;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

    int* d_flag;
    checkCudaError(cudaMalloc(&d_flag, sizeof(int)), "Unable to allocate flag value");
    
    int* d_rep;
    checkCudaError(cudaMalloc(&d_rep, numVert * sizeof(int)), "Unable to allocate rep array");

    unsigned char* d_flags;
    // Allocate memory on the GPU
    checkCudaError(cudaMalloc((void**)&d_flags, sizeof(unsigned char) * numVert), "cudaMalloc");

    // Set the allocated memory to zero
    checkCudaError(cudaMemset(d_flags, 0, sizeof(unsigned char) * numVert), "cudaMemset");

    // Allocate device output array and num selected
    int     *d_out            = NULL;
    int     *d_num_selected_out   = NULL;
    _g_allocator_.DeviceAllocate((void**)&d_out, sizeof(int) * numVert);
    _g_allocator_.DeviceAllocate((void**)&d_num_selected_out, sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    initialise<<<numBlocks, numThreads>>>(d_rep, numVert);
    checkCudaError(cudaDeviceSynchronize(), "Error in launching initialise kernel");

    int flag = 1;
    int iteration = 0;

    const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
    const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

    while(flag) {
        flag = 0;
        iteration++;
        checkCudaError(cudaMemcpy(d_flag, &flag, sizeof(int),cudaMemcpyHostToDevice), "Unable to copy the flag to device");

        hooking<<<numBlocks_hooking, numThreads>>> (numEdges, edge_u, edge_v, d_rep, d_flag, iteration);
        checkCudaError(cudaDeviceSynchronize(), "Error in launching hooking kernel");
        
        // #ifdef DEBUG
        //     cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost);
        //     // Printing the data
        //     std::cout << "\niteration num : "<< iteration << std::endl;
        //     std::cout << "d_rep : ";
        //     for (int i = 0; i < numVert; i++) {
        //         std::cout << host_rep[i] << " ";
        //     }
        //     std::cout << std::endl;
        // #endif

        for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
            short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);
            checkCudaError(cudaDeviceSynchronize(), "Error in launching short_cutting kernel");
        }

        checkCudaError(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), 
            "Unable to copy back flag to host");
    }

    select_rep<<<numBlocks_hooking, numThreads>>>(d_rep, d_flags, numVert);
    checkCudaError(cudaDeviceSynchronize(), "Error in launching select_rep kernel");

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_rep, d_flags, d_out, d_num_selected_out, numVert);
    _g_allocator_.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_rep, d_flags, d_out, d_num_selected_out, numVert);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    add_function_time("CC", duration);

    checkCudaError(cudaFree(d_flag), "Failed to free flag");
    checkCudaError(cudaFree(d_rep), "Failed to free d_rep");

    int h_num;
    cudaMemcpy(&h_num, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);

    return {d_out, h_num};
}
