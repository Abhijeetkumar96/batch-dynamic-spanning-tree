#include <iostream>
#include <vector>

#include "super_graph/super_graph.cuh"
#include "hash_table/HashTable.cuh"

#include "PR-RST/rootedSpanningTreePR.cuh"
#include "PR-RST/pr_rst_util.cuh"

#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"

#include "cuda_bfs/cuda_bfs.cuh"

// #define DEBUG

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

// //Combining two keys
__device__ __forceinline__
uint64_t combine_keys(uint32_t key1, uint32_t key2) 
{
    uint64_t combined_key = key1;
    combined_key = (combined_key << 32) | key2;
    return combined_key;
}

template <typename T>
__device__ 
void swap_32(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__global__
void combine_edges(
    int* d_super_graph_u,
    int* d_super_graph_v,
    uint64_t* d_super_graph,
    long num_edges) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_edges) {
        d_super_graph[tid] = combine_keys(d_super_graph_u[tid], d_super_graph_v[tid]);
    }
}

__global__
void superGraphKernel(
    keyValues* hashtable, 
    uint64_t* d_edge_list,
    int* rep_array, 
    int* d_repMap, 
    int *superGraph_u, 
    int *superGraph_v,
    long num_edges,
    int* d_counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_edges) {

        // get the actual edges from d_edge_list
        uint64_t edge = d_edge_list[tid];
        
        uint32_t d_nst_u = edge >> 32;
        uint32_t d_nst_v = edge & 0xFFFFFFFF;

        superGraph_u[tid] = d_repMap[rep_array[d_nst_u]];
        superGraph_v[tid] = d_repMap[rep_array[d_nst_v]];

        int value1 = d_nst_u;
        int value2 = d_nst_v;

        if(rep_array[d_nst_u] == rep_array[d_nst_v]) {
            return;
        }
        // uncomment to count the number of cross edges
        // atomicAdd(d_counter, 1);

        //insert into hashtable here only
        int key1 = superGraph_u[tid];
        int key2 = superGraph_v[tid];
        
        #ifdef DEBUG
            printf("For tid = %d, nst_u = %d, nst_v = %d, value1 = %d, value2 = %d, key1 = %d, key2 = %d\n", tid, d_nst_u, d_nst_v, value1, value2, key1, key2);
        #endif

        if(key1 > key2) {
            swap_32(key1, key2);
            swap_32(value1, value2);
        }
        #ifdef DEBUG
            printf("\n\n");
            printf("For tid = %d, value1 = %d, value2 = %d, key1 = %d, key2 = %d\n", tid, value1, value2, key1, key2);
            printf("\n\n");
        #endif
        unsigned long long key = combine_keys(key1, key2);
        uint64_t slot = hash(key);

        while(1)
        {
            unsigned long long prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if(prev == kEmpty || prev == key)
            {
                hashtable[slot].val1 = value1;
                hashtable[slot].val2 = value2;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }

    }
}

//Lookup keys in the hashtable, and return the values
__global__
void gpu_hashtable_lookup(keyValues* hashtable, int* d_parent, int* d_unique_rep, int* edge_u, int* parent_u, unsigned int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
        int key1 = tid;
        int key2 = d_parent[tid];
        bool swapped = false;
        
        if(key1 > key2) {
            swap_32(key1, key2);
            swapped = true;
        }

        if(key1 == key2) {
            edge_u[tid]   = d_unique_rep[key1];
            parent_u[tid] = d_unique_rep[key2];
            return;
        }

        unsigned long long new_key = combine_keys(key1, key2);

        #ifdef DEBUG
            printf("\nFor key1 = %d, key2 = %d, combined_key = %lu\n", key1, key2, new_key);
        #endif

        uint64_t slot = hash(new_key);

        while(1)
        {
            if(hashtable[slot].key == new_key)
            {
                if(swapped) {
                    edge_u[tid] = hashtable[slot].val2;
                    parent_u[tid] = hashtable[slot].val1;
                } else {
                    edge_u[tid] = hashtable[slot].val1;
                    parent_u[tid] = hashtable[slot].val2;
                }
                return;
            }

            if(hashtable[slot].key == kEmpty)
            {
                // printf("\nFor key1 = %d, key2 = %d, value1 = %d, value2 = %d, no value found\n",key1, key2, hashtable[slot].val1, hashtable[slot].val2);   
                edge_u[tid] = -1;
                parent_u[tid] = -1;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void get_replacement_edges(dynamic_tree_manager& tree_ds, PR_RST& pr_resource_mag, const int& unique_rep_count) {

    //-------------------------- Assign local pointers ----------------------------
    keyValues* pHashTable   =   tree_ds.pHashTable;

    uint64_t* d_edge_list   =   tree_ds.d_updated_edge_list;
    uint64_t* d_super_graph =   tree_ds.d_edge_list;
    
    int* d_rep              =   tree_ds.d_parent;                                                                              
    int* d_rep_map          =   tree_ds.d_rep_map; 
    int* d_unique_rep       =   tree_ds.d_unique_rep;                                            
    int* parent_u           =   pr_resource_mag.d_parent_u;
    int* edge_u             =   pr_resource_mag.d_edge_u;
    
    int* d_super_graph_u    =   tree_ds.d_super_graph_u;
    int* d_super_graph_v    =   tree_ds.d_super_graph_v;
    
    int* d_new_super_graph_u = tree_ds.d_new_super_graph_u;
    int* d_new_super_graph_v = tree_ds.d_new_super_graph_v;

    int* unique_super_graph_edges = tree_ds.super_graph_edges;
    
    long num_edges = tree_ds.num_edges; 
    //------------------------------------------------------------------------------
    
    int* d_counter;
    cudaMallocHost(&d_counter, sizeof(int));
    *d_counter = 0;

    long numThreads = 1024;
    long numBlocks = (num_edges + numThreads - 1) / numThreads;

    auto start = std::chrono::high_resolution_clock::now();
    // create superGraph
    superGraphKernel<<<numBlocks, numThreads>>>(
        pHashTable, 
        d_edge_list, 
        d_rep, 
        d_rep_map, 
        d_super_graph_u,
        d_super_graph_v,
        num_edges,
        d_counter
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    // std::cout << "Total number of cross-edges: " << *d_counter << std::endl;
    
    // g_verbose = true;

    if(g_verbose) {
        std::cout << "printing from get_replacement_edges func:\n";
        DisplayDeviceEdgeList(d_super_graph_u, d_super_graph_v, num_edges);
    }
    
    // find unique edges
    // d_edge_list contains the new set of edges now and unique_super_graph_edges is the count
    // remove selfloops and duplicates

    remove_self_loops_duplicates(
        d_super_graph_u, 
        d_super_graph_v, 
        num_edges, 
        d_super_graph, 
        tree_ds.d_flags, 
        unique_super_graph_edges,
        d_new_super_graph_u, 
        d_new_super_graph_v);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Super graph Creation", duration);

    if(*unique_super_graph_edges < 1) {
        std::cerr << "No cross edges found to connect the graphs.\n";
        return;
    }
    
    g_verbose = false;
    
    #ifdef DEBUG
        std::cout << "Unique edges after removing self loops and dups: " << *unique_super_graph_edges << std::endl;
        DisplayDeviceEdgeList(d_new_super_graph_u, d_new_super_graph_v, *unique_super_graph_edges);
    #endif

    // Apply any parallel rooted spanning tree algorithm to get replacement edge.
    int* super_parent = RootedSpanningTree(d_new_super_graph_u, d_new_super_graph_v, pr_resource_mag.num_vert, *unique_super_graph_edges);

    // int* super_parent = cuda_BFS(d_new_super_graph_u, d_new_super_graph_v, pr_resource_mag.num_vert, *unique_super_graph_edges);

    // h_size is super_graph parent array size
    int h_size = pr_resource_mag.num_vert;
    // std::cout << "h_size (a.k.a. num_vert from pr_resource_mag): " << h_size << std::endl;
    g_verbose = true;
    
    if(g_verbose) {
        std::cout << "super_graph parent array:\n";
        print_device_array(super_parent, h_size);
    }

    // g_verbose = false;

    numBlocks = (h_size + numThreads - 1) / numThreads;

    start = std::chrono::high_resolution_clock::now();
    // get original edges
    gpu_hashtable_lookup<<<numBlocks, numThreads>>>(
        pHashTable,
        super_parent, 
        d_unique_rep,
        edge_u, 
        parent_u, 
        h_size
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after gpu_hashtable_lookup");

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Hash Table look up", duration);

    if(g_verbose) {
        std::cout << "edge_u:\n";
        print_device_array(edge_u, h_size);

        std::cout << "parent_u:\n";
        print_device_array(parent_u, h_size);
    }
}