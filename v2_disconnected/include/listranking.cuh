#ifndef LISTRANKING_H
#define LISTRANKING_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

void CudaSimpleListRank(int *devNext, int *devRank, int N, int *notAllDone, int *devNotAllDone, unsigned long long *devRankNext);

#endif // LISTRANKING_H
