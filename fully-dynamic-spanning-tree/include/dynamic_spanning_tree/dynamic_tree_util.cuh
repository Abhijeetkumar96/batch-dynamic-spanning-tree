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

#include "hash_table/HashTable.cuh"

class dynamic_tree_manager {
public:

    // host variables
    int num_vert = 0;
    long num_edges = 0;
    int delete_batch_size = 0;
    int tree_edge_count = 0;
    int root = -1;
    int* parent_array; // Pointer to dynamically allocated array
    std::vector<uint64_t> edges_to_delete;

    std::vector<int> original_rep;

    // device variables
    int* d_parent = nullptr;
    int* new_parent = nullptr;
    int* d_org_parent = nullptr;
    int* d_rep = nullptr;
    int* d_unique_rep = nullptr;
    int* d_rep_map = nullptr;

    // hooking-shortcutting params
    // ----------------------------------//
    int* d_componentParent = nullptr;
    int* d_parentEdge = nullptr;
    int* d_actual_roots = nullptr;
    int* d_super_tree_roots = nullptr;

    // these two are boolean variables indicating where to consider or not
    unsigned char* d_roots_flag = nullptr;
    unsigned char* d_cross_edges_flag = nullptr;
    //----------------------------------//

    // superGraph params
    // ----------------------------------//
    int* d_super_graph_u = nullptr;
    int* d_super_graph_v = nullptr;

    int* d_new_super_graph_u = nullptr;
    int* d_new_super_graph_v = nullptr;

    int* super_graph_edges = NULL;
    // ----------------------------------//

    uint64_t* d_edges_to_delete = nullptr;
    uint64_t* d_edges_to_insert = nullptr;
    uint64_t* d_super_tree_list = nullptr;

    // d_edge_list is the original edge_list
    uint64_t* d_edge_list = nullptr;
    // d_updated_edge_list is the new edgelist after deleting the edges
    uint64_t* d_updated_edge_list = nullptr;

    unsigned char* d_flags = NULL;
    void *d_temp_storage = NULL;

    keyValues* pHashTable = nullptr;
    
    std::unordered_map<std::string, double> function_times;

    dynamic_tree_manager(std::vector<int>& parent, const std::string& delete_filename, const std::vector<uint64_t>& edge_list, int root);
    void mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list);
    void update_existing_ds();
    void read_delete_batch(const std::string& delete_filename, std::vector<int>& parent);
    void create_hashtable_();
    void destroy_hashtable_();

    ~dynamic_tree_manager();
};

#endif // DYNAMIC_TREE_UTIL_H