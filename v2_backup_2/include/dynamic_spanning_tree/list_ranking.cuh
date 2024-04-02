#ifndef LIST_RANKING_H
#define LIST_RANKING_H

#include "common/cuda_utility.cuh"

class ListRanking {
public:
    int *devSum;
    int *devSublistHead;
    int *devSublistId;
    int *devLast;
    int *devNext;
    int s, N;

    ListRanking(int N) : N(N) {
        // Calculate 's' based on 'N'
        if (N >= 100000) {
            s = sqrt(N) * 1.6; // Adjust based on GPU; this example is for a GTX 980.
        } else {
            s = N / 100;
        }

        if (s == 0) {
            s = 1;
        }

        allocateMemory();
    }

    ~ListRanking() {
        cudaFree(devSum);
        cudaFree(devSublistHead);
        cudaFree(devSublistId);
        cudaFree(devLast);
        cudaFree(devNext);
    }

private:
    void allocateMemory() {
        CUDA_CHECK(cudaMalloc((void**)&devSum, sizeof(int) * (s + 1)),          "Failed to allocate memory for devSum");
        CUDA_CHECK(cudaMalloc((void**)&devSublistHead, sizeof(int) * (s + 1)),  "Failed to allocate memory for devSublistHead");
        CUDA_CHECK(cudaMalloc((void**)&devSublistId, sizeof(int) * N),          "Failed to allocate memory for devSublistId");
        CUDA_CHECK(cudaMalloc((void**)&devLast, sizeof(int) * (s + 1)),         "Failed to allocate memory for devLast");
        CUDA_CHECK(cudaMalloc((void**)&devNext, sizeof(int) * N),               "Failed to allocate memory for devNext");
    }
};

void cuda_list_rank(int N, int head, const int *devNextSrc, int *devRank, ListRanking* lrParam);

#endif