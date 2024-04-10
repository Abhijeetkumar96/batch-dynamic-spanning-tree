#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda.h>
#include <cub/cub.cuh>

#include "cuda_bfs/cuda_bfs.cuh"

// #define DEBUG

using namespace std;

__global__
void computesegments(long* d_nodes, int* d_edges, long d_m, long d_n, long d_nodefrontier_size , int* d_nodefrontier , long* d_segments){

  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_nodefrontier_size){
    int node = d_nodefrontier[id];
    long start = d_nodes[node];
    long end = (node == d_n - 1) ? d_m : d_nodes[node + 1];
    d_segments[id] = end - start;
  }

}

__global__
void computerank(long* d_rank , long* d_seg , long* d_segments , long d_edgefrontier_size){
  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    d_rank[id] = id - d_segments[d_seg[id]];
  }
}

__global__
void computeedgefrontier(int* d_edgefrontier ,int* d_nodefrontier, long d_edgefrontier_size , long* d_rank , long* d_seg , long* d_nodes , int* d_edges){
  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    long seg = d_seg[id];
    long rank = d_rank[id];
    d_edgefrontier[id] = d_edges[d_nodes[d_nodefrontier[seg]] + rank];
  }
}


__global__
void markvisited(long d_edgefrontier_size , int level , int* d_edgefrontier , int* d_distance , long* d_seg){
  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    d_seg[id] = (-1 == atomicCAS(d_distance + d_edgefrontier[id], -1, level));
  }
}


__global__
void computenodefrontier(int* d_nodefrontier , int* d_edgefrontier , long* d_seg , long d_nodefrontier_size){
  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_nodefrontier_size){
    d_nodefrontier[id] = d_edgefrontier[d_seg[id]];
  }
}


__global__ void custom_lbs(long *a, long n_a , long *b , long n_b){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i< n_b){
      long l = 1, r = n_a;
      while(l<r){
          long mid = (l+r)/2;
          if(a[mid]<= i){
              l = mid+1;
          }else{
              r = mid;
          }
      }
      b[i] = l-1;
  }
}

template <typename T1, typename T2>
__global__
void print_a(T1* arr, T2 n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        for(T2 i = 0; i < n; ++i) {
            printf("%d ", arr[i]);
        }
    }
}

void adam_polak_bfs(int n, long m, long* d_nodes, int* d_edges) {

    #ifdef DEBUG
        std::cout << "n: " << n <<" m : " << m << std::endl;
        std::cout << "d_nodes array;\n";
        print_a<<<1,1>>>(d_nodes, n);
        cudaDeviceSynchronize();
        std::cout << "d_edges array;\n";
        print_a<<<1,1>>>(d_edges, m);
        cudaDeviceSynchronize();
    #endif
  
    int* d_nodefrontier, *d_edgefrontier,* d_distance;
    long* d_rank;
    
    cudaMalloc(&d_nodefrontier, n * sizeof(int));
    cudaMalloc(&d_edgefrontier, m * sizeof(int));
    cudaMalloc(&d_rank, max(static_cast<long>(n), m) * sizeof(long));
    cudaMalloc(&d_distance, n * sizeof(int));

    long* d_segments, *d_seg, *d_seg1;

    cudaMalloc(&d_segments, (max(static_cast<long> (n), m) + 2) * sizeof(long));
    cudaMalloc(&d_seg, (max(static_cast<long>(n), m) + 2) * sizeof(long));
    cudaMalloc(&d_seg1,(max(static_cast<long> (n), m) + 2) * sizeof(long));

    //temp storage for cub scan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_segments, d_segments, max(static_cast<long> (n) , m) + 2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    void *d_temp_storage1 = NULL;
    size_t temp_storage_bytes1 = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_seg, d_seg1, max(static_cast<long> (n), m)+2);
    cudaMalloc(&d_temp_storage1, temp_storage_bytes1);

    cudaMemset(d_distance, -1, n * sizeof(int));
    cudaMemset(d_distance, 0, sizeof(int));
    cudaMemset(d_nodefrontier, 0, sizeof(int));

    int* level;
    long* h_nodefrontier_size;
    long* h_edgefrontier_size;
    
    cudaMallocHost((void**)&h_nodefrontier_size, sizeof(long));
    cudaMallocHost((void**)&level, sizeof(int));
    cudaMallocHost((void**)&h_edgefrontier_size, sizeof(long));
    
    h_nodefrontier_size[0] = 1;
    h_edgefrontier_size[0] = 0;

    while(h_nodefrontier_size[0] > 0) {

        level[0]++;

        computesegments<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_nodes, d_edges, m, n, h_nodefrontier_size[0], d_nodefrontier, d_segments);
        cudaDeviceSynchronize();

        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_segments, d_segments, h_nodefrontier_size[0]+1);

        cudaMemcpy(&h_edgefrontier_size[0], d_segments + h_nodefrontier_size[0], sizeof(long), cudaMemcpyDeviceToHost);

        custom_lbs<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_segments, h_nodefrontier_size[0],d_seg , h_edgefrontier_size[0]);
        cudaDeviceSynchronize();
          
        computerank<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_rank, d_seg , d_segments, h_edgefrontier_size[0]);
        cudaDeviceSynchronize();

        computeedgefrontier<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_edgefrontier, d_nodefrontier, h_edgefrontier_size[0], d_rank, d_seg , d_nodes, d_edges);
        cudaDeviceSynchronize();

        markvisited<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(h_edgefrontier_size[0], level[0], d_edgefrontier, d_distance, d_seg);
        cudaDeviceSynchronize();

        cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_seg, d_seg1, h_edgefrontier_size[0]+1);

        cudaMemcpy(&h_nodefrontier_size[0], d_seg1 + h_edgefrontier_size[0], sizeof(long), cudaMemcpyDeviceToHost);

        custom_lbs<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_seg1 , h_edgefrontier_size[0], d_seg , h_nodefrontier_size[0]);
        cudaDeviceSynchronize();

        computenodefrontier<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_nodefrontier, d_edgefrontier, d_seg , h_nodefrontier_size[0]);
        cudaDeviceSynchronize();
    }

        
        // print d_distance to output_updated.txt
        int* h_distance = (int*)malloc(n * sizeof(int));
        cudaMemcpy(h_distance, d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        #ifdef DEBUG
          std::cout << "numNodes: " << n << "\n";
            for(int i = 0; i < n; i++)
                cout << h_distance[i] << " ";
            std::cout << std::endl;
        #endif
}