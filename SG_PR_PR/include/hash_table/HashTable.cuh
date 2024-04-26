#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include<iostream>
#include <cuda_runtime.h>

struct keyValues {
    unsigned long long key;
    uint32_t val1;
    uint32_t val2;
};

// Constants
const uint64_t kHashTableCapacity = 128 * 1024 * 1024;
const uint64_t kEmpty = 0xffffffffffffffff;

// Function declarations
keyValues* create_hashtable();
void destroy_hashtable(keyValues* pHashTable);

#endif // HASH_TABLE_CUH
