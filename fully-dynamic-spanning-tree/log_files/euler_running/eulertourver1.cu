#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include "headers/eulerTour.h"
#include "headers/mytimer.h"

using namespace std;

__device__ int cuAbs(int i) { return i < 0 ? -i : i; }

__global__ void copy_to_device(int* edges_from_input , int* edges_to_input , int* edges_from , int* edges_to , int N){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < N-1){
        edges_from[thid + N - 1] = edges_to[thid] = edges_to_input[thid];
        edges_to[thid + N - 1] = edges_from[thid] = edges_from_input[thid];
    }
}

__global__ void init_index(int* index , int E){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < E){
        index[i] = i;
    }
}

__global__ void init_next(int* next , int E){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < E){
        next[i] = -1;
    }
}

__global__ void fill_nextandfirst(int* edges_from , int* edges_to , int* index , int* next , int* first , int E){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < E){
        int f = edges_from[index[thid]];
        //int t = edges_to[index[thid]];

        if (thid == 0) {
          first[f] = index[thid];
          return;
        }

        int pf = edges_from[index[thid - 1]];
        //int pt = edges_to[index[thid - 1]];

        if (f != pf) {
          first[f] = index[thid];
        } else {
          next[index[thid - 1]] = index[thid];
        } 
    }
}

__global__ void fill_succ(int* edges_from , int* next , int* first , int* succ , int E){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < E){
        int revEdge = (thid + E / 2) % E;
        if (next[revEdge] == -1) {
          succ[thid] = first[edges_from[revEdge]];
        } else {
          succ[thid] = next[revEdge];
        }
    }
}


__global__ void init_devNext(int* devNext , int* devNextSrc , int N , int head){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N){
      devNext[i] = devNextSrc[i];
      if (devNextSrc[i] == head)
        devNext[i] = -1;
  }
}

__global__ void init_devRank(int* devRank , int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N){
      devRank[i] = 0;
  }
}


__global__ void split(int* devNext, int* devSublistHead, int N, int s , int head){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < s){
      curandState state;
      curand_init(123, i, 0, &state);

      int p = i * (N / s);
      int q = min(p + N / s, N);

      int splitter;
      do {
        splitter = (cuAbs(curand(&state)) % (q - p)) + p;
      } while (devNext[splitter] == -1);

      devSublistHead[i + 1] = devNext[splitter];
      devNext[splitter] = -i - 2; // To avoid confusion with -1

      if (i == 0) {
        devSublistHead[0] = head;
      }
  }
}

__global__ void updateall(int* devNext, int* devRank, int* devSublistHead, int* devSublistId, int* devLast, int* devSum , int s){
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid < s + 1){
      int current = devSublistHead[thid];
      int counter = 0;

      while (current >= 0) {
        
        devRank[current] = counter++;

        int n = devNext[current];

        if (n < 0) {
          devSum[thid] = counter;
          devLast[thid] = current;
        }

        devSublistId[current] = thid;
        current = n;
      }
  }
}


__global__ void kernelf(int head , int s, int* devNext , int* devLast , int* devSum){
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid == 0){
      int tmpSum = 0;
      int current = head;
      int currentSublist = 0;
      for (int i = 0; i <= s; i++) {
        tmpSum += devSum[currentSublist];
        devSum[currentSublist] = tmpSum - devSum[currentSublist];

        current = devLast[currentSublist];
        currentSublist = -devNext[current] - 1;
      }
  }
}



__global__ void kernelf1(int* devSum , int* devRank , int* devSublistId , int N){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < N){
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
    }
}

__global__ void fill_ness(int* d_parent , int* edges_from_input , int* edges_to_input , int N , int root){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < N){
      if (thid == root)return;
    int afterRoot = thid > root;
    edges_from_input[thid - afterRoot] = thid;
    edges_to_input[thid - afterRoot] = d_parent[thid];
    }
}

__global__ void getMET(int* rank_to_output , int* firstOccurrence , int* lastOccurrence , int* mod_euler_tour , int N , int* edges_from , int* edges_to , int* d_level , int root){
  int thid = blockIdx.x * blockDim.x + threadIdx.x;
  if(thid < 2*N-2){
      int u = edges_from[thid];
      int v = edges_to[thid];
      if(d_level[u] < d_level[v]){
          firstOccurrence[v] = rank_to_output[thid]+1;
          mod_euler_tour[rank_to_output[thid]+1] = 1;
      }
      else{
          lastOccurrence[u] = rank_to_output[thid]+1;
          mod_euler_tour[rank_to_output[thid]+1] = 0;
      }
      
      if(thid==0){
        mod_euler_tour[0] = 1;
        mod_euler_tour[2*N-1] = 0;
        firstOccurrence[root] = 0;
        lastOccurrence[root] = 2*N-1;
      }

  }
}


void cuda_list_rank(int N, int head, int *devNextSrc, int *devRank){

  mytimer mt3{};

  int s;
  if (N >= 100000) {
    s = sqrt(N) * 1.6; // Based on experimental results for GTX 980.
  } else
    s = N / 100;
  if (s == 0) s = 1;

  int *devNext;
  cudaMalloc((void **)&devNext, sizeof(int) * (N));
  
  int num_threads = 1024;
  int num_blocks = (N + 1024 - 1) / 1024;
  init_devNext<<<num_blocks , num_threads>>>(devNext  , devNextSrc , N , head);
  cudaDeviceSynchronize();

  init_devRank<<<num_blocks , num_threads>>>(devRank , N);
  cudaDeviceSynchronize();

  int *devSum;
  cudaMalloc((void **)&devSum, sizeof(int) * (s + 1));
  int *devSublistHead;
  cudaMalloc((void **)&devSublistHead, sizeof(int) * (s + 1));
  int *devSublistId;
  cudaMalloc((void **)&devSublistId, sizeof(int) * N);
  int *devLast;
  cudaMalloc((void **)&devLast, sizeof(int) * (s + 1));

  mt3.timetaken_reset("alloc" , 0 );

  num_blocks = (s + 1024 - 1) / 1024;
  split<<<num_blocks , num_threads>>>(devNext, devSublistHead, N, s , head);
  cudaDeviceSynchronize();


  num_blocks = (s + 1 + 1024 - 1) / 1024;
  updateall<<<num_blocks , num_threads>>>(devNext, devRank, devSublistHead, devSublistId, devLast, devSum , s);
  cudaDeviceSynchronize();

  kernelf<<<1,1>>>(head , s, devNext , devLast , devSum);
  cudaDeviceSynchronize();


  num_blocks = (N + 1024 - 1) / 1024;
  kernelf1<<<num_blocks , num_threads>>>(devSum , devRank ,devSublistId , N);
  cudaDeviceSynchronize();

  mt3.timetaken_reset("step 6 : list_rank" , 1 );

  cudaFree(devNext);
  cudaFree(devSum);
  cudaFree(devSublistHead);
  cudaFree(devSublistId);
  cudaFree(devLast);

}



void cuda_euler_tour(int N , int root , int* edges_from_input, int* edges_to_input , int* rank_to_output , int* firstOccurrence , int* lastOccurrence , int* mod_euler_tour , int* d_level){


    mytimer mt2{};

  void *d_temp_storage1 = NULL;
  size_t temp_storage_bytes1 = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, mod_euler_tour, mod_euler_tour, 2*N);
  cudaMalloc(&d_temp_storage1, temp_storage_bytes1);

    int E = N*2 - 2;

    int *edges_to;
    int *edges_to_copy;
    int *edges_from;
    int *edges_from_copy;
    cudaMalloc((void **)&edges_to, sizeof(int) * E);
    cudaMalloc((void **)&edges_from, sizeof(int) * E);
    cudaMalloc((void **)&edges_to_copy, sizeof(int) * E);
    cudaMalloc((void **)&edges_from_copy, sizeof(int) * E);
    int *index;
    cudaMalloc((void **)&index, sizeof(int) * E);
    int *index1;
    cudaMalloc((void **)&index1, sizeof(int) * E);
    int *index2;
    cudaMalloc((void **)&index2, sizeof(int) * E);

    int num_threads = 1024;
    int num_blocks = (N-1 + num_threads - 1) / num_threads;
    copy_to_device<<<num_blocks , num_threads>>>(edges_from_input , edges_to_input , edges_from , edges_to , N);
    cudaDeviceSynchronize();

    num_threads = 1024;
    num_blocks = (E + num_threads - 1) / num_threads;
    init_index<<<num_blocks , num_threads>>>(index , E);
    cudaDeviceSynchronize();

    int *h_index1;
    cudaMallocHost((void **)&h_index1, sizeof(int) * E);
    cudaMemcpy(h_index1, index, sizeof(int) * E, cudaMemcpyDeviceToHost);

    
    int *next;
    cudaMalloc((void **)&next, sizeof(int) * E);
    int* succ;
    cudaMalloc((void **)&succ, sizeof(int) * E);
    int* head;
    cudaMallocHost((void **)&head, sizeof(int) * 1);
    int* first;
    cudaMalloc((void **)&first, sizeof(int) * N);


    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, edges_to , edges_to_copy , index , index ,E);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    mt2.timetaken_reset("step 1 : sort" , 0 );

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, edges_to , edges_to_copy , index , index1 ,E);
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, edges_to , edges_to_copy , edges_from , edges_from_copy ,E);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, edges_from_copy , edges_to_copy , index1 , index2 , E);

    

    cudaFree(d_temp_storage);
    cudaFree(index);
    cudaFree(index1);
    cudaFree(edges_from_copy);
    cudaFree(edges_to_copy);



    init_next<<<num_blocks , num_threads>>>(next , E);
    cudaDeviceSynchronize();

    

    fill_nextandfirst<<<num_blocks , num_threads>>>(edges_from , edges_to , index2 , next , first , E);
    cudaDeviceSynchronize();

    
    cudaFree(index2);

    

    fill_succ<<<num_blocks , num_threads>>>(edges_from , next , first , succ , E);
    cudaDeviceSynchronize();

    mt2.timetaken_reset("step 3 : fill_succ" , 1 );

    cudaMemcpy(head, first + root, sizeof(int), cudaMemcpyDeviceToHost);

    
    cudaFree(next);
    cudaFree(first);

    cuda_list_rank(E , *head , succ , rank_to_output);

    num_blocks = (2*N - 2 + 1024 - 1) / 1024;

    mt2.timetaken_reset("step 4 : list_rank" , 0 );

    getMET<<<num_blocks , num_threads>>>(rank_to_output , firstOccurrence , lastOccurrence , mod_euler_tour , N , edges_from , edges_to , d_level , root);
    cudaDeviceSynchronize();

    cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, mod_euler_tour, mod_euler_tour , 2*N);

    mt2.timetaken_reset("step 5 : getMET" , 1 );
    
    cudaFree(succ);
    cudaFree(edges_to);
    cudaFree(edges_from);
}


void getModifiedEulerTour(int *d_parent , int* d_level , int n , int* mod_euler_tour , int* firstOccurrence , int*  lastOccurrence , int root){

  mytimer mt1{};
  
  int* edges_from_input;
  int* edges_to_input;
  cudaMalloc((void **)&edges_from_input, sizeof(int) * (n-1));
  cudaMalloc((void **)&edges_to_input, sizeof(int) * (n-1));
  int* rank_to_output;
  cudaMalloc((void **)&rank_to_output, sizeof(int) * n*2);


  int num_threads = 1024;
  int num_blocks = (n + num_threads - 1) / num_threads;

  mt1.timetaken_reset("alloc" , 0 );

  fill_ness<<<num_blocks , num_threads>>>(d_parent , edges_from_input , edges_to_input , n , root);
  cudaDeviceSynchronize();

  mt1.timetaken_reset("step 1 : fill_ness " , 1 );

  cuda_euler_tour(n , root , edges_from_input , edges_to_input , rank_to_output , firstOccurrence , lastOccurrence , mod_euler_tour , d_level);

  cudaFree(edges_from_input);
  cudaFree(edges_to_input);
  cudaFree(rank_to_output);

}





// int main(){
  
//   freopen("amazon_parent.txt", "r", stdin);
//   int n;
//   cin>>n;
//   int* parent;
//   cudaMallocHost((void **)&parent, sizeof(int) * n);
//   int* level;
//   cudaMallocHost((void **)&level, sizeof(int) * n);
//   for(int i=0;i<n;i++){
//     cin>>parent[i];
//   }

//   ifstream file("output_valid_amazon.txt");
//   for(int i=0;i<n;i++){
//     file>>level[i];
//   }
//   file.close();

//   int* d_level;
//   cudaMalloc((void **)&d_level, sizeof(int) * n);
//   cudaMemcpy(d_level, level, sizeof(int) * n, cudaMemcpyHostToDevice);

//   int* d_parent;
//   cudaMalloc((void **)&d_parent, sizeof(int) * n);
//   cudaMemcpy(d_parent, parent, sizeof(int) * n, cudaMemcpyHostToDevice);

//   int* d_mod_euler_tour;
//   cudaMalloc((void **)&d_mod_euler_tour, sizeof(int) * n*2);

//   int* d_firstOccurrence;
//   cudaMalloc((void **)&d_firstOccurrence, sizeof(int) * n);

//   int* d_lastOccurrence;
//   cudaMalloc((void **)&d_lastOccurrence, sizeof(int) * n);

//   getModifiedEulerTour(d_parent , d_level , n , d_mod_euler_tour , d_firstOccurrence , d_lastOccurrence , 0);

    
//   return 0;

// }