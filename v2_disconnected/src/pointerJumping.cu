#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>
#include "pointerJumping.cuh"

__global__ 
void p_jump_kernel(int n, int *d_componentNumber) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid < n) {
        if(d_componentNumber[tid] != tid) {
            while (d_componentNumber[tid] != d_componentNumber[d_componentNumber[tid]])
            {
                d_componentNumber[tid] = d_componentNumber[d_componentNumber[tid]];
            }
        }
    }
}

void pointer_jumping(int* d_next, int n) {
    
    // calculate the optimal number of threads and blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto parallel_start = std::chrono::high_resolution_clock::now();  
    
    p_jump_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_next);
    cudaDeviceSynchronize();

    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(parallel_end - parallel_start).count();
    std::cout << "Total time for parallel pointer jumping : " << parallel_duration << " microseconds (" << n << " number of keys)\n";
}