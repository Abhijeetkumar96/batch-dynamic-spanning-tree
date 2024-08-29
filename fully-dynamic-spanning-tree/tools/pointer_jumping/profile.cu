#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

void print(std::vector<int>& arr)
{
    for(auto i : arr)
        std::cout<<i<<" ";
    std::cout<<std::endl;
}

bool verify(std::vector<int>& arr1, std::vector<int>& arr2)
{
    for(int i = 0; i< arr1.size(); ++i)
    {
        if(arr1[i] != arr2[i])
        {
            std::cout<<"Verification failed at "<<i;
            return 0;
        }
    }
    return 1;
}

__global__
void pointer_jumping_kernel(int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {

        if(next[tid] != tid)
        {
            next[tid] = next[next[tid]];
        }
    }
}

void pointerJumping(std::vector<int>& next){
    int n = next.size();
    int size = n * sizeof(int);
    int *d_next;
    //Allocate GPU memory
    cudaMalloc((void**)&d_next, size);
    cudaMemcpy(d_next, next.data(), size, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto parallel_start = std::chrono::high_resolution_clock::now();  
    for (int j = 0; j < std::ceil(std::log2(n)); ++j)
    {
        pointer_jumping_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, n);
        cudaDeviceSynchronize();
    }
    // cudaDeviceSynchronize();
    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(parallel_end - parallel_start).count();
    printf("Total time for my parallel pointer jumping : %ld microseconds (%d number of keys)\n", parallel_duration, n);  

    cudaFree(d_next);
}

std::vector<int> create_next_array(unsigned int N) 
{
    std::vector<int> list(N);

    // Initialize the vector
    for (unsigned int i = 0; i < N; ++i) {
        list[i] = i;
    }

    // Use the current time as a seed for the random number generator
    std::srand(unsigned(std::time(0)));

    // Shuffle the elements of the vector
    std::mt19937 engine(std::time(0));
    std::shuffle(list.begin(), list.end(), engine);

    // Print the shuffled vector
    // for (auto i : list)
    //     std::cout << i << ' ';
    // std::cout << '\n';

    std::vector<int> next(N);
    for(int i = 0; i<= N - 2; ++i)
    {
        next[list[i]] = list[i+1];
    }
    next[list[N-1]] = list[N-1];

    // for(auto i : next)
    //     std::cout<<i<<" ";
    // std::cout<<std::endl;

    return next;
}


int main() {
    int start_range = 10000000; // 10M
    int end_range = 70000000; // 70M

    cudaFree(0);

    int seed = 10000000; //10M
    for(int k = start_range; k<=end_range; k += seed) {
        std::vector<int> next = create_next_array(k);
        pointerJumping(next);
        std::cout << std::endl;
    }

    return 0;
}