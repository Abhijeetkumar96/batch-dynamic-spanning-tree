//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

//---------------------------------------------------------------------
// CUDA Libraries
//---------------------------------------------------------------------
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// CSR Specific funtions & utilities
//---------------------------------------------------------------------
#include "cuda_bfs/cuda_csr.cuh"
#include "common/cuda_utility.cuh"

//---------------------------------------------------------------------
// CUDA Kernels
//---------------------------------------------------------------------

// Duplicate creation kernel
__global__ 
void dup_creation_Kernel(int* d_u, int* d_v, int* d_u_out, int* d_v_out, long size) {
    
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_u_out[idx] = d_u[idx];
        d_v_out[idx] = d_v[idx];
        d_u_out[idx + size] = d_v[idx]; 
        d_v_out[idx + size] = d_u[idx];
    }
}

void create_duplicate(int* d_u, int* d_v, int* d_u_out, int* d_v_out, long size) {
    long maxThreadsPerBlock = 1024;
    long blocksPerGrid = (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    dup_creation_Kernel<<<blocksPerGrid, maxThreadsPerBlock>>>(d_u, d_v, d_u_out, d_v_out, size);
}


// CSR starts
__global__
void cal_offset(int no_of_vertices, long no_of_edges, int* dir_U, long* offset) {

    // based on the assumption that the graph is connected and graph size is > 2
    long tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid == 0) 
        offset[tid] = 0;

    if(tid == no_of_vertices) 
        offset[tid] = no_of_edges;

    if(tid < no_of_edges - 1) {
        if(dir_U[tid] != dir_U[tid + 1]) {

            int v = dir_U[tid + 1];
            offset[v] = tid + 1;
        }
    }
}   

void gpu_csr(
    cub::DoubleBuffer<int>& d_keys, 
    cub::DoubleBuffer<int>& d_values, 
    long num_items, 
    const int numvert, 
    long* d_vertices) 
{
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    // determine temp_storage
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate d_temp_storage");
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);

    long maxThreadsPerBlock = 1024;
    long numBlocks = (num_items + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
        
    // Launch the kernel with the myStream passed as the last argument
    cal_offset<<<numBlocks, maxThreadsPerBlock>>>(numvert, num_items, d_keys.Current(), d_vertices);
    CUDA_CHECK(cudaGetLastError(), "cal_offset kernel launch failed");
}

// ====[ End of gpu_csr Code ]====