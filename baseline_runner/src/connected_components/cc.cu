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

#include "connected_components/cc.cuh"

// #define DEBUG

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

int cc(int* edge_u, int* edge_v, int numVert, long numEdges, std::string filename) {

    bool _g_verbose = false;
    if(_g_verbose) {
        std::vector<int> host_rep(numVert);
    
        // write the updated the edges to file
        std::vector<int> h_edge_u(numEdges);
        std::vector<int> h_edge_v(numEdges);
    
        checkCudaError(cudaMemcpy(h_edge_u.data(), edge_u, numEdges * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy back");
    
        checkCudaError(cudaMemcpy(h_edge_v.data(), edge_v, numEdges * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy back");
    
        std::string output_path = "/raid/graphwork/spanning_tree_datasets/bridges_deleted/";
        std::string output_filename = output_path + cc_get_filename(filename) + ".txt";
        std::ofstream outFile(output_filename);
    
        outFile << numVert << " " << 2 * numEdges << "\n";
        
        for(long i = 0; i < numEdges; ++i) {
            outFile << h_edge_u[i] << " " << h_edge_v[i] << "\n";
            outFile << h_edge_v[i] << " " << h_edge_u[i] << "\n";
        }
    
        // #ifdef DEBUG
        //     std::cout << "numEdges from cc: " << numEdges << " \n";
        //     print_list<<<1,1>>>(edge_u, edge_v, numEdges);
        //     cudaDeviceSynchronize();
        // #endif
    }

    const long numThreads = 1024;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

    int* d_flag;
    checkCudaError(cudaMalloc(&d_flag, sizeof(int)), "Unable to allocate flag value");

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

        for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
            short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);
            err = cudaGetLastError();
            checkCudaError(err, "Error in launching short_cutting kernel");
        }

        checkCudaError(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), 
            "Unable to copy back flag to host");
    }

    return 1;
}
