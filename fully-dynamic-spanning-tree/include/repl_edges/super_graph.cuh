#ifndef SUPER_GRAPH_H
#define SUPER_GRAPH_H

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "repl_edges/repl_edges.cuh"

//64 bit Murmur2 hash
__device__ __forceinline__
uint64_t hash(const uint64_t key);

//Combining two keys
__device__ __forceinline__
uint64_t combine_keys(uint32_t key1, uint32_t key2);

// Lookup keys in the hashtable, and return the values
__global__
void gpu_hashtable_lookup(keyValues* hashtable, int* d_parent, int* d_unique_rep, int* edge_u, int* parent_u, unsigned int size);

/**
 * Computes replacement edges for the dynamic spanning tree.
 * This function creates a superGraph and derives a rooted tree based on the unique representatives
 * of the graph's vertices. It then updates the dynamic tree manager and resource manager with
 * the new edges that replace those removed during updates to the spanning tree.
 *
 * @param tree_ds Reference to the dynamic tree manager, holding the current state of the dynamic spanning tree.
 * @param resource_mag Reference to the resource manager, responsible for managing graph resources and properties.
 * @param unique_rep_count The count of unique representatives, used in calculating the replacement edges.
 * @param is_deletion set to true for a deletion operation, false for an insertion operation.
 */
void super_graph(dynamic_tree_manager& tree_ds, 
    REP_EDGES& rep_edge_mag,  
    bool is_deletion);

#endif // SUPER_GRAPH_H