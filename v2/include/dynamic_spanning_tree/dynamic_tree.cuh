#ifndef DYNAMIC_TREE_H
#define DYNAMIC_TREE_H

#include "euler_tour.cuh"
#include "dynamic_tree_util.cuh"

struct keyValues
{
    unsigned long long key;
    uint32_t val1;
    uint32_t val2;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;

const uint64_t kEmpty = 0xffffffffffffffff;

keyValues* create_hashtable();
void destroy_hashtable(keyValues* pHashTable);
// void superGraph(const std::vector<int> nst_u, const std::vector<int> nst_v, const std::vector<int> rep_array, const std::vector<int> repMap, const std::vector<int> unique_rep, const std::vector<int> parent, thrust::host_vector<int> h_first, thrust::host_vector<int> h_last, int src);

void repair_spanning_tree(const std::vector<int>& roots, dynamic_tree_manager& tree_ds, EulerianTour& euler_tour);

#endif // DYNAMIC_TREE_H