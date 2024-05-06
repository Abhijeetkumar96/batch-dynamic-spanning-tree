#include "hash_table/HashTable.cuh"
#include "common/cuda_utility.cuh"

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
    CUDA_CHECK(cudaFree(pHashTable), "Failed to destroy hash_table");
}
