#include <cuda_runtime.h>

#include "common/cuda_utility.cuh"
#include "common/Timer.hpp"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "repl_edges/euler_tour.cuh"
#include "repl_edges/repl_edges.cuh"
#include "repl_edges/super_graph.cuh"
#include "repl_edges/hooking_shortcutting.cuh"

// #define DEBUG

__global__ 
void HOOKING(
    long edges, 
    uint64_t* d_edge_list,
    int *rep, 
    int *componentParent, 
    bool isMaxIteration, 
    int *c_flag) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < edges) {
        
        uint64_t i = d_edge_list[tid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int rep_u = rep[u];
        int rep_v = rep[v];

        if(rep_u != rep_v) {
            // 2 different components
            *c_flag = true;
            if(isMaxIteration) {
                componentParent[min(rep_u, rep_v)] = max(rep_u, rep_v);
            }
            else {
                componentParent[max(rep_u, rep_v)] = min(rep_u, rep_v);
            }
        }
    }
}

__global__ 
void UPDATE_REP_PARENT(int nodes, int *componentParent, int *rep) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < nodes) {
        if(rep[tid] == tid && componentParent[tid] != -1) {
            rep[tid] = componentParent[tid];
        }
    }
}

/**
 * Needs to be executed before UPDATE_REP_PARENT
 * @d_parentEdge : d_parentEdge[i] --> idx of the edge which connects ith tree to parent of ith tree
*/
__global__ 
void STORE_TRANS_EDGES(
    int edges,
    int *rep, 
    uint64_t* d_edge_list,
    int *componentParent, 
    int *d_parentEdge) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < edges) {
       
        uint64_t i = d_edge_list[tid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int rep_u = rep[u];
        int rep_v = rep[v];

        // printf("u = %d, v = %d , rep_u = %d, rep_v = %d \n", u, v, rep_u, rep_v);
        if( rep_v == componentParent[rep_u]){
            // u is the representative of the tree
            // v belongs to the parent tree of u

            d_parentEdge[rep_u] = tid;
        }

        if( rep_u == componentParent[rep_v]) {
            d_parentEdge[rep_v] = tid;
        }   
    }
}

__global__ 
void SHORTCUTTING(int nodes, int *rep, int *flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nodes)
    {
        int prevValue = rep[tid];
        rep[tid] = rep[rep[tid]];
        if (prevValue != rep[tid])
        {
            *flag = 1;
        }
    }
}

__global__
void GET_CROSS_EDGES(int nodes, int *d_parentEdge, unsigned char *d_cross_edges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;;
    if(tid < nodes) {
        if(d_parentEdge[tid] != -1) {
            d_cross_edges[d_parentEdge[tid]] = 1;
        }
    }
}

__global__
void GET_ROOTS(int nodes, int *rep, unsigned char *roots) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;;
    if(tid < nodes) {
        if(rep[tid] == tid) {
            roots[tid] = 1;
        }
    }
}

void hooking(
    int nodes, long edges,
    uint64_t* d_edge_list,
    int* d_rep,                     // rep[i] --> representative of the tree of which i is a part
    int* d_componentParent,         // componentParent[i] =rep of parent tree of the ith tree
    int* d_parentEdge,
    unsigned char* d_cross_edges,   // d_cross_edges[i] = 1 --> this is valid edge that connects 2 trees
    unsigned char* d_roots,         // is this node a root
    int* c_flag, 
    int* c_shortcutFlag) {

    #ifdef DEBUG
        std::cout << "Printing from hooking function:" << std::endl;

        std::cout << "nodes: " << nodes << ", edges: " << edges << std::endl;

        std::cout << "rep array:" << std::endl;
        print_device_array(d_rep, nodes);

        std::cout << "Edges input to hooking: " << std::endl;
        print_device_edge_list(d_edge_list, edges);
    #endif

    int num_threads = 1024;

    int num_blocks_edges = (edges + num_threads - 1) / num_threads;
    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    int flag = 1;
    int shortcutFlag = 1;
    bool maxIteration = true;

    while(flag) {

        flag = false;
        CUDA_CHECK(cudaMemcpy(c_flag, &flag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy flag to device");
        HOOKING <<<num_blocks_edges, num_threads>>> (
            edges,
            d_edge_list,
            d_rep,
            d_componentParent,
            maxIteration,
            c_flag);

        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after HOOKING");
        
        CUDA_CHECK(cudaMemcpy(&flag, c_flag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy flag back to device");
        maxIteration = !maxIteration;

        // !!! This should be done before updating
        STORE_TRANS_EDGES<<<num_blocks_edges, num_threads>>> (
            edges,
            d_rep,
            d_edge_list,
            d_componentParent,
            d_parentEdge
        );
        
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after STORE_TRANS_EDGES");

        // rep[representative] = representative of its parent
        UPDATE_REP_PARENT<<<num_blocks_vert, num_threads>>> (
            nodes,
            d_componentParent,
            d_rep);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after UPDATE_REP_PARENT");
        
        shortcutFlag = true;
        while(shortcutFlag) {
            shortcutFlag = false;
            CUDA_CHECK(cudaMemcpy(c_shortcutFlag, &shortcutFlag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy shortcutFlag to device");
            SHORTCUTTING <<<num_blocks_vert, num_threads >>> (nodes, d_rep, c_shortcutFlag);
            CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after SHORTCUTTING kernel");
            CUDA_CHECK(cudaMemcpy(&shortcutFlag, c_shortcutFlag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back shortcutFlag to host");
        }
    }

    #ifdef DEBUG
        std::cout << "Printing Final Rep array:" << std::endl;
        print_device_array(d_rep, nodes);
        std::cout << std::endl;

        std::cout << "Printing Final Rep array:" << std::endl;
        print_device_array(d_parentEdge, nodes);
        std::cout << std::endl;
    #endif

    GET_ROOTS<<<num_blocks_vert, num_threads>>>(
        nodes,
        d_rep,
        d_roots);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after GET_ROOTS kernel");

    GET_CROSS_EDGES<<<num_blocks_vert, num_threads>>>(
        nodes, 
        d_parentEdge, 
        d_cross_edges);
            
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after GET_CROSS_EDGES kernel");

    #ifdef DEBUG
        std::cout << "Roots" << std::endl;
        print_device_array(d_roots, nodes);

        std::cout<<"Cross Edges" << std::endl;
        print_device_array(d_cross_edges, edges);
    #endif
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
void swap_32_(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__global__
void insert_hashTable(
    keyValues* hashtable, 
    uint64_t* d_edge_list,
    int* rep_array, 
    int* d_repMap,
    long num_edges) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if(tid == 0) {
    //     printf("Printing from insert_hashTable:\n");
    //     for(int i = 0; i < 22; ++i) {
    //         printf("rep_array[%d]: %d\n", i, rep_array[i]);
    //     }
    // }

    if(tid < num_edges) {

        // get the actual edges from d_edge_list
        uint64_t edge = d_edge_list[tid];
        
        uint32_t d_nst_u = edge >> 32;
        uint32_t d_nst_v = edge & 0xFFFFFFFF;

        int value1 = d_nst_u;
        int value2 = d_nst_v;

        //insert into hashtable here only
        int key1 = d_repMap[rep_array[d_nst_u]];
        int key2 = d_repMap[rep_array[d_nst_v]];
        
        if(key1 > key2) {
            swap_32_(key1, key2);
            swap_32_(value1, value2);
        }
        #ifdef DEBUG
            printf("\n\n");
            printf("For tid = %d, value1 = %d, value2 = %d, key1 = %d, key2 = %d\n", tid, value1, value2, key1, key2);
            printf("\n\n");
        #endif

        unsigned long long key = combine_keys(key1, key2);
        d_edge_list[tid] = key;
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

__global__
void update_roots_mapping(int* d_roots, int* d_rep, int* d_rep_map, int roots_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < roots_count) {
        int root = d_roots[tid];
        int rep_root = d_rep[root];
        d_roots[tid] = d_rep_map[rep_root];
    }
}

void hooking_shortcutting(dynamic_tree_manager& tree_ds, 
    REP_EDGES& rep_edge_mag,
    bool is_deletion) {

    //-------------------------- Assign local pointers ----------------------------
    keyValues* pHashTable   =   tree_ds.pHashTable;

    uint64_t* d_edge_list       =   nullptr;                    // input:  edge_list
    uint64_t* d_super_tree      =   tree_ds.d_edge_list;        // output: all the cross_edges

    int* d_rep                  =   tree_ds.d_parent;           // input
    int* d_rep_map              =   tree_ds.d_rep_map;          // input
    int* d_unique_rep           =   tree_ds.d_unique_rep;       // input                                 
    int* d_componentParent      =   tree_ds.d_componentParent;
    int* d_parentEdge           =   tree_ds.d_parentEdge;
    int* d_actual_roots         =   tree_ds.d_actual_roots;
    int* d_super_tree_roots     =   tree_ds.d_super_tree_roots;

    unsigned char* d_cross_edges_flag =   tree_ds.d_cross_edges_flag;
    unsigned char* d_roots_flag       =   tree_ds.d_roots_flag;

    int* parent_u           =   rep_edge_mag.d_parent_u;    // output: the actual replacement_edges_u
    int* edge_u             =   rep_edge_mag.d_edge_u;      // output: the actual replacement_edges_v
    
    int nodes            =   tree_ds.num_vert;
    long edges           =   tree_ds.num_edges; 
    //------------------------------------------------------------------------------

    // reset 
    CUDA_CHECK(cudaMemset(d_roots_flag,       0, nodes * sizeof(unsigned char)), "Failed to memset d_roots");
    CUDA_CHECK(cudaMemset(d_parentEdge,      -1, nodes * sizeof(int)),           "Failed to memset d_parentEdge");
    CUDA_CHECK(cudaMemset(d_cross_edges_flag, 0, edges * sizeof(unsigned char)), "Failed to memset d_cross_edges");


    if(is_deletion)
        d_edge_list = tree_ds.d_updated_edge_list;
    else {
        d_edge_list = tree_ds.d_edges_to_insert;
        edges = tree_ds.delete_batch_size;
    }

    #ifdef DEBUG
        std::cout << "d_unique_rep: " << std::endl;
        print_device_array(d_unique_rep, nodes);

        std::cout<<"d_rep_map" << std::endl;
        print_device_array(d_rep_map, nodes);
    #endif

    int* d_rep_hook;
    CUDA_CHECK(cudaMalloc(&d_rep_hook, sizeof(int) * nodes), "Failed to allocate memory for d_rep_hook");
    CUDA_CHECK(cudaMemcpy(d_rep_hook, d_rep, sizeof(int) * nodes, cudaMemcpyDeviceToDevice), "Failed to copy d_rep_org to device");

    int *c_flag;
    CUDA_CHECK(cudaMallocHost((void **)&c_flag, sizeof(int)), "Failed to allocate memory for c_flag");
    int *c_shortcutFlag;
    CUDA_CHECK(cudaMallocHost((void **)&c_shortcutFlag, sizeof(int)), "Failed to allocate memory for c_shortcutFlag");

    auto start = std::chrono::high_resolution_clock::now();
    hooking(
        nodes, edges, 
        d_edge_list, 
        d_rep_hook, 
        d_componentParent, 
        d_parentEdge, 
        d_cross_edges_flag, 
        d_roots_flag,
        c_flag,
        c_shortcutFlag);

    long num_cross_edges = edges;

    // find actual cross_edges
    select_flagged(d_edge_list, d_super_tree, d_cross_edges_flag, num_cross_edges);

    int num_roots = nodes;
    // find actual roots
    select_flagged(d_actual_roots, d_super_tree_roots, d_roots_flag, num_roots);

    int numThreads = 256;
    long numBlocks = (num_roots + numThreads - 1) / numThreads;

    update_roots_mapping<<<numBlocks, numThreads>>>(d_super_tree_roots, d_rep, d_rep_map, num_roots);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_roots_mapping");

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("HS: Find Repl edges", duration);

    *tree_ds.super_graph_edges = static_cast<int>(num_cross_edges);

    std::cout << "Number of roots: " << num_roots << std::endl;
    std::cout << "Number of cross-edges: " << num_cross_edges << std::endl;

    if(num_cross_edges < 1)
        return;

    #ifdef DEBUG
        std::cout << "All cross-edges:" << std::endl;
        print_device_edge_list(d_super_tree, num_cross_edges);

        std::cout << "selected roots:" << std::endl;
        print_device_array(d_super_tree_roots, num_roots);
    #endif

    numThreads = 1024;
    numBlocks = (num_cross_edges + numThreads - 1) / numThreads;

    start = std::chrono::high_resolution_clock::now();
    // add to hash_table
    insert_hashTable<<<numBlocks, numThreads>>>(
        pHashTable, 
        d_super_tree, 
        d_rep, 
        d_rep_map, 
        num_cross_edges
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();

    add_function_time("Hashtable insertion", duration);

    std::cout << "Insertion into hashtable over." << std::endl;

    int h_size = rep_edge_mag.num_vert;
    std::cout << "num of superComponents: " << h_size << std::endl;

    Euler_Tour euler(h_size, num_cross_edges, num_roots);

    start = std::chrono::high_resolution_clock::now();
    
    // Apply eulerianTour algorithm to root an unrooted tree to get replacement edge.
    cuda_euler_tour(
        d_super_tree,       // edgelist
        h_size,             // num_vertices
        num_cross_edges,    // num_edges
        d_super_tree_roots, // roots
        num_roots,          // count of roots
        euler);              

    int* super_parent = euler.d_parent;

    numBlocks = (h_size + numThreads - 1) / numThreads;

    // retrieve actual edges from hashTable
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

    add_function_time("ET: Orientation", duration);

    #ifdef DEBUG
        std::cout << "num of Replacement edges: " << num_cross_edges << std::endl;
        std::cout << "Replacement edges:" << std::endl;
        DisplayDeviceEdgeList(edge_u, parent_u, h_size);
    #endif
}