//Create a hash table. For linear probing, this is just an array of keyValues
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>
#include <cuda.h>

struct keyValues
{
    unsigned long long key;
    uint32_t val1;
    uint32_t val2;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;

const uint64_t kEmpty = 0xffffffffffffffff;

keyValues* create_hashtable()
{
    keyValues* hashtable;
    cudaMalloc(&hashtable, sizeof(keyValues) * kHashTableCapacity);

    //Initialize hash table to empty
    static_assert(kEmpty == 0xffffffffffffffff, "memset expected kEmpty=0xffffffffffffffff");
    cudaMemset(hashtable, 0xff, sizeof(keyValues) * kHashTableCapacity);

    return hashtable;
}

void destroy_hashtable(keyValues* pHashTable)
{
    cudaFree(pHashTable);
}

//64 bit Murmur2 hash
__device__ __forceinline__
uint64_t hash(const uint64_t key)
{
    const uint32_t seed = 0x9747b28c;
    const uint64_t m = 0xc6a4a7935bd1e995LLU; // A large prime number
    const int r = 47;
  
    uint64_t h = seed ^ (8 * m);

    uint64_t k = key;
  
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  
    // Finalization
    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h & (kHashTableCapacity - 1); //mask to ensure it falls within table
}

//Combining two keys
__host__ __device__ __forceinline__
uint64_t combine_keys(uint32_t key1, uint32_t key2) 
{
    uint64_t combined_key = key1;
    combined_key = (combined_key << 32) | key2;
    return combined_key;
}

__global__
void superGraphKernel(keyValues* hashtable, uint64_t* d_edge_list, int *rep_array, int *d_repMap, int *superGraph_u, int *superGraph_v, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size)
    {
        // printf("\nFor tid = %d, nst_u = %d, nst_v = %d, rep_u = %d, rep_v = %d, mapped_rep_u = %d, mapped_rev_v = %d ", tid, d_nst_u[tid], d_nst_v[tid], d_nst_u[tid], d_nst_v[tid], d_repMap[rep_array[d_nst_u[tid]]], d_repMap[rep_array[d_nst_v[tid]]]);
        uint64_t edge = d_edge_list[tid];
        uint32_t d_nst_u = edge >> 32;
        uint32_t d_nst_v = edge & 0xFFFFFFFF;
        superGraph_u[tid] = d_repMap[rep_array[d_nst_u]];
        superGraph_v[tid] = d_repMap[rep_array[d_nst_v]];

        //insert into hashtable here only
        unsigned long long key = combine_keys(superGraph_u[tid], superGraph_v[tid]);
        uint64_t slot = hash(key);

        while(1)
        {
            unsigned long long prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if(prev == kEmpty || prev == key)
            {
                hashtable[slot].val1 = d_nst_u;
                hashtable[slot].val2 = d_nst_v;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }

    }
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ 
void gpu_iterate_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t* kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity) 
    {
        if (pHashTable[threadid].key != kEmpty) 
        {
            uint32_t value = pHashTable[threadid].value;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    uint32_t* device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
    cudaFree(device_num_kvs);

    return kvs;
}

int main(void) {

    std::vector<int> nst_u = {3, 4, 6};
    std::vector<int> nst_v = {4, 5, 8};

    std::vector<uint64_t> h_edge_list;

    // combine nst_u and nst_v to create a unique representation
    // and push to h_edge_list
    for(int i = 0; i < nst_u.size(); i++)
    {
        h_edge_list.push_back(combine_keys(nst_u[i], nst_v[i]));
    }
    
    std::vector rep_array = {0, 1, 2, 1, 0, 0, 0, 0, 9, 9};
    std::vector<int> repMap = {0, 1, 2, 0, 0, 0, 0, 0, 0, 3};
    std::vector<int> unique_rep = {0, 1, 2, 9};

    keyValues* pHashTable = create_hashtable();
    unsigned int n = unique_rep.size();
    unsigned int size = nst_u.size();
    // Copy the input data to the device
    thrust::device_vector<uint64_t> d_edge_list = h_edge_list;
    thrust::device_vector<int> d_rep_array = rep_array;
    thrust::device_vector<int> d_repMap = repMap;

    // Create output vectors on the device
    thrust::device_vector<int> d_superGraph_u(size);
    thrust::device_vector<int> d_superGraph_v(size);

    dim3 blockDim = 1024;
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    // Launch the kernel
    superGraphKernel<<<gridDim, blockDim>>>(
        pHashTable,
        thrust::raw_pointer_cast(d_edge_list.data()),
        thrust::raw_pointer_cast(d_rep_array.data()),
        thrust::raw_pointer_cast(d_repMap.data()),
        thrust::raw_pointer_cast(d_superGraph_u.data()),
        thrust::raw_pointer_cast(d_superGraph_v.data()),
        size);

    cudaDeviceSynchronize();

    // Get all the key-values from the hash table
    std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

    // print the key-values
    for (const auto& kv : kvs) 
    {
        std::cout << "key: " << kv.key << ", value: " << kv.value << std::endl;
    }

}