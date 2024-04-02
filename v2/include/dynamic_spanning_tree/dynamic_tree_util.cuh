/******************************************************************************
* Functionality: Memory Management
* Handles allocation, deallocation, and initialization of all variables
 ******************************************************************************/

#ifndef DYNAMIC_TREE_UTIL_H
#define DYNAMIC_TREE_UTIL_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

class dynamic_tree_manager {
public:

    // host variables
    int num_vert = 0;
    long num_edges = 0;
    int delete_batch_size = 0;
    std::vector<uint64_t> edges_to_delete;

    // device variables
    int* d_parent = nullptr;
    int* d_org_parent = nullptr;
    int* d_rep = nullptr;
    int* d_unique_rep = nullptr;
    int* d_rep_map = nullptr;
    uint64_t* d_edges_to_delete = nullptr;

    // d_edge_list is the original edge_list
    uint64_t* d_edge_list = nullptr;
    // d_updated_edge_list is the new edgelist after deleting the edges
    uint64_t* d_updated_edge_list = nullptr;

    void mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list);
    void update_existing_ds();
    void read_delete_batch(const std::string& delete_filename);

    ~dynamic_tree_manager();
};

#endif // DYNAMIC_TREE_UTIL_H