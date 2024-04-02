#include <cuda_runtime.h>

#include "pr_rst/pr_rst.cuh"
#include "pr_rst/rootedSpanningTreePR.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "common/cuda_utility.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "dynamic_spanning_tree/dynamic_tree_util.cuh"

//Create a hash table. For linear probing, this is just an array of keyValues
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
    cudaFree(pHashTable);
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

//Combining two keys
__device__ __forceinline__
uint64_t combine_keys(uint32_t key1, uint32_t key2) 
{
    uint64_t combined_key = key1;
    combined_key = (combined_key << 32) | key2;
    return combined_key;
}

//Lookup keys in the hashtable, and return the values
__global__
void gpu_hashtable_lookup(keyValues* hashtable, int* d_parent, uint32_t* d_val1, uint32_t* d_val2, unsigned int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > 0 && tid < size)
    {
        uint32_t key1 = tid;
        uint32_t key2 = d_parent[tid];

        unsigned long long new_key = combine_keys(key1, key2);
        uint64_t slot = hash(new_key);

        while(1)
        {
            if(hashtable[slot].key == new_key)
            {
                d_val1[tid] = hashtable[slot].val1;
                d_val2[tid] = hashtable[slot].val2;
                // printf("\nFor key1 = %u, key2 = %u, value1 = %u, value2 = %u\n",key1, key2, hashtable[slot].val1, hashtable[slot].val2);
                return;
            }

            if(hashtable[slot].key == kEmpty)
            {
                d_val1[tid] = kEmpty;
                d_val2[tid] = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

__global__
void superGraphKernel(
	keyValues* hashtable, 
	uint64_t* d_edge_list, 
	int *rep_array, 
	int *d_rep_map, 
	uint64_t* d_super_graph, 
	long num_edges) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < num_edges) {
	    
        uint64_t t = d_edge_list[tid];

        uint32_t value1 = t >> 32;  // Extract higher 32 bits
        uint32_t value2 = t & 0xFFFFFFFF; // Extract lower 32 bits

        uint32_t x = d_rep_map[rep_array[value1]];
        uint32_t y = d_rep_map[rep_array[value2]];

        unsigned long long key = combine_keys(x, y);
        // uint64_t key = combine_keys(x, y);
        d_super_graph[tid] = key;
        
        //insert into hashtable here only
        uint64_t slot = hash(key);

        while(1) {
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
void update_rep_map(int* d_unique_rep, int* d_rep_map, int unique_rep_count) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < unique_rep_count) {
        printf("tid: %d, unique_rep: %d, rep_map pos: %d\n", tid, d_unique_rep[tid], tid);
    	d_rep_map[d_unique_rep[tid]] = tid;
    }
}

void create_super_graph(dynamic_tree_manager& tree_ds, const int& unique_rep_count) {

	keyValues* pHashTable = create_hashtable();
	
	uint64_t* d_edge_list = tree_ds.d_updated_edge_list;
	uint64_t* d_super_graph = tree_ds.d_edge_list;

	int* d_rep = tree_ds.d_parent;
	int* d_rep_map = tree_ds.d_rep_map;

	long num_edges = tree_ds.num_edges;
	
	long numThreads = 1024;
	long numBlocks = (num_edges + numThreads - 1) / numThreads;

	superGraphKernel<<<numBlocks, numThreads>>>(
		pHashTable, 
		d_edge_list, 
		d_rep, 
		d_rep_map, 
		d_super_graph, 
		num_edges
	);

	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

	std::cout << "remapped edges:\n";
	print_device_edge_list(d_super_graph, num_edges);

	long unique_super_graph_edges = 0;
	
	// find unique edges
	// d_edge_list contains the new set of edges now and unique_super_graph_edges is the count
	find_unique(d_super_graph, d_edge_list, num_edges, unique_super_graph_edges);
	std::cout << "unique_super_graph_edges: " << unique_super_graph_edges << "\n";
	std::cout << "Super graph Edges:\n";
	print_device_edge_list(d_edge_list + 1, unique_super_graph_edges - 1);

	// RST_Resource_manager pr_resources(unique_rep_count);
	
	// Apply any parallel rooted spanning tree algorithm to get replacement edge.
	// RootedSpanningTree(
	// pr_resources, 
	// d_edge_list + 1,				// edge_list + 1 to avoid one 0
	// unique_rep_count, 				// this many vertices are there in superGraph
	// unique_super_graph_edges - 1);	// edges

	// int* new_parent = pr_resources.d_new_parent_ptr;
	// print_device_array(new_parent, tree_ds.delete_batch_size + 1);
}

void repair_spanning_tree(const std::vector<int>& roots, dynamic_tree_manager& tree_ds, EulerianTour& euler_tour) {

	int* d_rep = tree_ds.d_parent;
	int* d_unique_rep = tree_ds.d_unique_rep;
	int* d_rep_map = tree_ds.d_rep_map;
	int num_vert = tree_ds.num_vert;

	std::cout << "parent array after deleting edges:\n";
	print_device_array(d_rep, num_vert);

	// 1. find eulerian tour
	cal_first_last(roots[0], tree_ds.d_org_parent, euler_tour);

	// 2. Do pointer jumping over parent array to update representative array.
	pointer_jumping(d_rep, tree_ds.num_vert);
	std::cout << "After doing pointer_jumping:\n";
	print_device_array(d_rep, num_vert);
	// 3. find unique in the d_rep array
	// send a copy of d_rep.
	int unique_rep_count = 0;
	find_unique(d_rep, d_unique_rep, tree_ds.num_vert, unique_rep_count);
	std::cout << "unique_rep_count: " << unique_rep_count << std::endl;
	
	std::cout << "d_unique_rep array:\n";
	print_device_array(d_unique_rep, unique_rep_count);

	int numThreads = 1024;
	int numBlocks = (unique_rep_count + numThreads - 1) / numThreads;
	
	// update rep_map
	update_rep_map<<<numBlocks, numThreads>>>(d_unique_rep, d_rep_map, unique_rep_count);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after update_rep_map");
	std::cout << "d_rep_map array:\n";
	print_device_array(d_rep_map, num_vert);
	
	// weed out self loops and duplicates
	// get the rooted spanning tree too
	create_super_graph(tree_ds, unique_rep_count);
}

