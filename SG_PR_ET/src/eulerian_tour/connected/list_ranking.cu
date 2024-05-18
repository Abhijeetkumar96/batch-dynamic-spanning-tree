#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "eulerian_tour/connected/list_ranking.cuh"

__device__ 
int cuAbs(int i) { 
  return i < 0 ? -i : i; 
}

__global__ 
void initDevNextKernel(int N, int head, const int *devNextSrc, int *devNext, int *devRank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        devNext[i] = devNextSrc[i];
        devRank[i] = 0;

        if (devNextSrc[i] == head)
            devNext[i] = -1;
    }
}

__global__ 
void initializeRankKernel(int *devRank, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        devRank[i] = 0;
    }
}

__global__ 
void assignSublistHeadsKernel(int N, int s, int head, int *devNext, int *devSublistHead) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s) {
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

__global__ 
void computeLocalRanksKernel(int s, const int *devSublistHead, const int *devNext, int *devRank, int *devSum, int *devLast, int *devSublistId) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < s + 1) {
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

__global__ 
void adjustGlobalRanksKernel(int head, int s, const int *devNext, const int *devLast, int *devSum) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < 1) { // This kernel is designed to run with a single thread
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

__global__ 
void updateRankKernel(const int *devSublistId, const int *devSum, int *devRank, int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < N) {
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
    }
}

void cuda_list_rank(int N, int head, const int *devNextSrc, int *devRank, ListRanking* lrParam) {
    
    // std::cout << "N: " << N << std::endl;
    
    int s = lrParam->s;
    int *devSum = lrParam->devSum;
    int *devSublistHead = lrParam->devSublistHead;
    int *devSublistId = lrParam->devSublistId;
    int *devLast = lrParam->devLast;
    int *devNext = lrParam->devNext;
    
    int numThreads = 1024; 
    int numBlocks = (N + numThreads - 1) / numThreads;

    initDevNextKernel<<<numBlocks, numThreads>>>(N, head, devNextSrc, devNext, devRank);
    
    numBlocks = (s +  numThreads - 1) /  numThreads;
    assignSublistHeadsKernel<<<numBlocks,  numThreads>>>(N, s, head, devNext, devSublistHead);

    numBlocks = (s + 1 +  numThreads - 1) /  numThreads;
    computeLocalRanksKernel<<<numBlocks,  numThreads>>>(s + 1, devSublistHead, devNext, devRank, devSum, devLast, devSublistId);
    adjustGlobalRanksKernel<<<1, 1>>>(head, s, devNext, devLast, devSum); // Single thread
    
    numBlocks = (N +  numThreads - 1) /  numThreads;
    updateRankKernel<<<numBlocks,  numThreads>>>(devSublistId, devSum, devRank, N);

    cudaMemcpy(&devRank[head], &N, sizeof(int), cudaMemcpyHostToDevice);


    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
}
