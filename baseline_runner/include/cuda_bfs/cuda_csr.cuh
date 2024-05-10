#ifndef CUDA_CSR_H
#define CUDA_CSR_H

#include <cub/cub.cuh>
#include <cuda_runtime.h>

void create_duplicate(int* d_u, int* d_v, int* d_u_out, int* d_v_out, long size);
void gpu_csr(cub::DoubleBuffer<int>& d_keys, cub::DoubleBuffer<int>& d_values, long num_items, const int numvert, long* d_vertices);

#endif // CUDA_CSR_H