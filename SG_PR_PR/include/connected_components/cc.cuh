#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

int cc(int* edge_u, int* edge_v, int numVert, long numEdges, std::string filename);

inline void checkCudaError(cudaError_t err, const std::string& msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif // CONNECTED_COMPONENTS_H