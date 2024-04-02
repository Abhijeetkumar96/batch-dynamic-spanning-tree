#ifndef LOCK_FREE_HASH_TABLE_H
#define LOCK_FREE_HASH_TABLE_H

#include <cuda_runtime.h>


struct keyValues {
    uint64_t key;
    uint32_t val1;
    uint32_t val2;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;

keyValues* create_hashtable() {
    keyValues* hashtable;
    CUDA_CHECK(cudaMalloc(&hashtable, sizeof(keyValues) * kHashTableCapacity), "Failed to allocate hashtable");
    CUDA_CHECK(cudaMemset(hashtable, 0xff, sizeof(keyValues) * kHashTableCapacity), "Failed to initialize hashtable");
    return hashtable;
}

void destroy_hashtable(keyValues* pHashTable) {
    CUDA_CHECK(cudaFree(pHashTable), "Failed to free hashtable");
}

#endif // LOCK_FREE_HASH_TABLE_H