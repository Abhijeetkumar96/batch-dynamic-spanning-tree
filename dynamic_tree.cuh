/******************************************************************************
* Functionality: Memory Management
* Handles allocation, deallocation, and initialization of all variables
 ******************************************************************************/

#ifndef DYNAMIC_TREE_H
#define DYNAMIC_TREE_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>


struct keyValues {
    uint64_t key;
    uint32_t val1;
    uint32_t val2;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;

keyValues* create_hashtable();
void destroy_hashtable(keyValues* pHashTable);

class dynamic_tree_manager {
public:

    void mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list);
    void update_existing_ds();
    void read_delete_batch(const std::string& delete_filename);

    ~dynamic_tree_manager();

    int num_vert = 0;
    long num_edges = 0;
    int delete_batch_size = 0;

    int* d_parent = nullptr;
    int* d_rep = nullptr;
    int* d_unique_rep = nullptr;
    uint64_t* d_edges_to_delete = nullptr;
    uint64_t* d_edge_list = nullptr;
    keyValues* pHashTable = nullptr;

    std::vector<uint64_t> edges_to_delete;
};

#endif // DYNAMIC_TREE_H