#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

void print(const std::vector<int> &arr);
void print(const thrust::device_vector<int> &arr);
void printArr(const std::vector<int> &parent,int vertices,int stride);
void printPR(const std::vector<int> &pr_arr,const std::vector<int> &pr_arr_size,int vertices,int log_2_size);