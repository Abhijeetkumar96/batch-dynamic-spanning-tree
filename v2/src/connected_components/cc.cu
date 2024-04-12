#include <set>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "connected_components/cc.cuh"

// Function to print available and total memory
void printMemoryInfo(const std::string& message) {
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
void print_(uint64_t* d_edges, long numEdges) {
    
    long tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid == 0) {
        
        for(long i = 0; i < numEdges; ++i) {
            uint64_t edge = d_edges[i];
            int edge_u = (int)edge & 0xFFFFFFFF;
            int edge_v = (int)(edge >> 32);

            printf("edge_u: %d, edge_v: %d\n", edge_u, edge_v);
        }
    }
}

__global__ 
void print_array(uint64_t* d_edges, long numEdges) {
    
    long tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid == 0) {
        
        for(long i = 0; i < numEdges; ++i) {
            printf("edge[%ld]: %llu\n", i, d_edges[i]);
        }
    }
}

void cc(int* edge_u, int* edge_v, int numVert, long numEdges) {

    std::vector<int> host_rep(numVert);

    // printMemoryInfo("after allocation is done");

    const long numThreads = 1024;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

    int* d_flag;
    checkCudaError(cudaMalloc(&d_flag, sizeof(int)), "Unable to allocate flag value");
    auto start = std::chrono::high_resolution_clock::now();
    int* d_rep;
    checkCudaError(cudaMalloc(&d_rep, numVert * sizeof(int)), "Unable to allocate rep array");

    initialise<<<numBlocks, numThreads>>>(d_rep, numVert);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Error in launching initialise kernel");

    int flag = 1;
    int iteration = 0;

    const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
    const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

    while(flag) {
        flag = 0;
        iteration++;
        checkCudaError(cudaMemcpy(d_flag, &flag, sizeof(int),cudaMemcpyHostToDevice), "Unable to copy the flag to device");

        hooking<<<numBlocks_hooking, numThreads>>> (numEdges, edge_u, edge_v, d_rep, d_flag, iteration);
        err = cudaGetLastError();
        checkCudaError(err, "Error in launching hooking kernel");
        
        #ifdef DEBUG
            cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost);
            // Printing the data
            std::cout << "\niteration num : "<< iteration << std::endl;
            std::cout << "d_rep : ";
            for (int i = 0; i < numVert; i++) {
                std::cout << host_rep[i] << " ";
            }
            std::cout << std::endl;
        #endif

        for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
            short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);
            err = cudaGetLastError();
            checkCudaError(err, "Error in launching short_cutting kernel");
        }

        checkCudaError(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back flag to host");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout <<"cc took " << duration << " ms." << std::endl;
    // std::cout <<"Number of iteration: " << iteration << std::endl;
    checkCudaError(cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back rep array");
    std::set<int> num_comp(host_rep.begin(), host_rep.end());

    std::cout <<"numComp = " << num_comp.size() << std::endl;

    checkCudaError(cudaFree(d_flag), "Failed to free flag");
    checkCudaError(cudaFree(d_rep), "Failed to free d_rep");
}
