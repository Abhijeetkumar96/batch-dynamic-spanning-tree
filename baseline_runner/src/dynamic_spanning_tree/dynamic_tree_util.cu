#include "common/cuda_utility.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"
#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "dynamic_spanning_tree/update_ds.cuh"

#include "hash_table/HashTable.cuh"

// #define DEBUG

using namespace cub;

void cpu_pointer_jumping(int* parent, int num_vert) {
    for (int i = 0; i < num_vert; ++i) {
        int root = i;
        // Find the root of the current element
        while (parent[root] != root) {
            root = parent[root];
        }
        // Path compression: update the parent of all elements along the path to point directly to the root
        int current = i;
        while (parent[current] != root) {
            int next = parent[current];
            parent[current] = root;
            current = next;
        }
    }
}

// Constructor
dynamic_tree_manager::dynamic_tree_manager(std::vector<int>& parent, const std::string& delete_filename, const std::vector<uint64_t>& edge_list, int _root) {

    num_vert = parent.size();
    num_edges = edge_list.size();
    root = _root;

    parent_array = new int[num_vert]; // Allocate memory for the array

    // Copy data from the input vector to the newly allocated array
    std::memcpy(parent_array, parent.data(), num_vert * sizeof(int));
    // std::cout << "Reading delete edges file\n";
    read_delete_batch(delete_filename, parent);
    // std::cout << "Reading completed.\n";
    
    // std::cout << "Allocating gpu memory\n";
    mem_alloc(parent, edge_list);
    // std::cout << "Allocation over.\n";

    // std::cout << "Updating data structure\n";
    update_existing_ds();
    std::cout << std::endl;
}

size_t AllocateTempStorage(void** d_temp_storage, long num_items) {
    size_t temp_storage_bytes = 0;
    size_t required_bytes = 0;

    // Determine the temporary storage requirement for DeviceRadixSort::SortPairs
    cub::DeviceRadixSort::SortPairs(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceScan::InclusiveSum
    cub::DeviceScan::InclusiveSum(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceSelect::Flagged
    cub::DeviceSelect::Flagged(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Allocate the maximum required temporary storage
    CUDA_CHECK(cudaMalloc(d_temp_storage, temp_storage_bytes), "cudaMalloc failed for temporary storage for CUB operations");

    return temp_storage_bytes;
}

void dynamic_tree_manager::mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list) {

    size_t size = parent.size() * sizeof(int);
    size_t delete_size = edges_to_delete.size() * sizeof(uint64_t);
    size_t edge_list_size = edge_list.size() * sizeof(uint64_t);
    
    // Allocate device memory

    pHashTable = create_hashtable();

    // allocate temp storage
    AllocateTempStorage(&d_temp_storage, 2 * edge_list.size());

    CUDA_CHECK(cudaMalloc(&d_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&new_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&d_org_parent, size), "Failed to allocate memory for d_org_parent");
    CUDA_CHECK(cudaMalloc(&d_unique_rep, size), "Failed to allocate memory for d_unique_rep");
    CUDA_CHECK(cudaMalloc(&d_rep_map, size), "Failed to allocate memory for d_rep_map");
    CUDA_CHECK(cudaMalloc(&d_edges_to_delete, delete_size), "Failed to allocate memory for edges to delete");
    
    // d_edge_list is the original edge_list
    CUDA_CHECK(cudaMalloc(&d_edge_list, edge_list_size), "Failed to allocate memory for input edge list");
    
    // d_updated_edge_list is the new edgelist after deleting the edges
    CUDA_CHECK(cudaMalloc(&d_updated_edge_list, edge_list_size), "Failed to allocate memory for input edge list");

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), size, cudaMemcpyHostToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_org_parent, d_parent, size,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(new_parent, d_parent, size,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_delete, edges_to_delete.data(), delete_size, cudaMemcpyHostToDevice), "Failed to copy edges to delete to device");
    CUDA_CHECK(cudaMemcpy(d_edge_list, edge_list.data(), edge_list_size, cudaMemcpyHostToDevice), "Failed to copy edge list to device");
    
    CUDA_CHECK(cudaMalloc((void **)&d_super_graph_u, num_edges * sizeof(int)), "Failed to allocate device memory for d_super_graph_u");
    CUDA_CHECK(cudaMalloc((void **)&d_super_graph_v, num_edges * sizeof(int)), "Failed to allocate device memory for d_super_graph_v");

    CUDA_CHECK(cudaMalloc((void **)&d_new_super_graph_u, num_edges * sizeof(int)), "Failed to allocate device memory for d_new_super_graph_u");
    CUDA_CHECK(cudaMalloc((void **)&d_new_super_graph_v, num_edges * sizeof(int)), "Failed to allocate device memory for d_new_super_graph_v");

    CUDA_CHECK(cudaMallocManaged((void**)&super_graph_edges, sizeof(int)),   "Failed to allocate d_num_selected_out");
    CUDA_CHECK(cudaMalloc((void**)&d_flags, num_edges * sizeof(unsigned char)), "Failed to allocate flag array");
}

void dynamic_tree_manager::read_delete_batch(const std::string& delete_filename, std::vector<int>& parent) {

    std::ifstream inputFile(delete_filename);
    if (!inputFile) {
        std::cerr << "Failed to open file: " << delete_filename << std::endl;
        return;
    }
    
    // n_edges: Number of edges to delete, including both tree and non-tree edges.
    int n_edges;
    inputFile >> n_edges;
    delete_batch_size = n_edges;
    uint32_t u, v;
    edges_to_delete.resize(n_edges);
    
    tree_edge_count = 0;
    
    // std::cout << "Reading " << n_edges << " edges from the file." << std::endl;

    for (int i = 0; i < n_edges; ++i) {
        inputFile >> u >> v;
        if(u > v) {
            // Ensures u is always less than v for consistent edge representation
            std::swap(u, v);
        }

        if(parent_array[u] == v or parent_array[v] == u) {
            tree_edge_count++;

            if(u == parent_array[v]) {
                parent_array[v] = v; // Disconnect the child from its parent

            } else if (v == parent_array[u]) {
                parent_array[u] = u;
            }
        }

        edges_to_delete[i] = ((uint64_t)(u) << 32 | v);
    }
    cpu_pointer_jumping(parent_array, num_vert);

    std::cout << "Number of deleted tree edges: " << tree_edge_count << std::endl;
    
    if(g_verbose) {

        // std::cout << "edges_to_delete array uint64_t:\n";

        // for(auto i : edges_to_delete)
        //     std::cout << i <<" ";
        // std::cout << std::endl;

        // std::cout << "edges_to_delete array:\n";
        // for(const auto &i : edges_to_delete)
        //     std::cout << (i >> 32) << " " << (i & 0xFFFFFFFF) << "\n";
        // std::cout << std::endl;
    }
}

void dynamic_tree_manager::update_existing_ds() {
	update_edgelist(
        d_parent,               // input -- 1
        num_vert,               // input -- 2
        d_edge_list,            // input -- 3
        d_updated_edge_list,    // output -- 4
        num_edges,              // output -- 5
        d_edges_to_delete,      // input -- 6
        delete_batch_size,      // input -- 7
        d_unique_rep,           // output -- 8
        unique_rep_count,       // output -- 9
        root);                  // input -- 10

    // now num_edges contains nonTreeEdges - parent_size - delete_batch count.

    CUDA_CHECK(cudaMemcpy(new_parent, d_parent, num_vert * sizeof(int), cudaMemcpyDeviceToDevice), 
        "Failed to copy d_parent to device");
}

dynamic_tree_manager::~dynamic_tree_manager() {
    delete[] parent_array;

    cudaFree(d_parent);
    cudaFree(d_unique_rep);
    cudaFree(d_edges_to_delete);
    cudaFree(d_edge_list);
    destroy_hashtable(pHashTable);
}

// ====[ End of update ds Code ]====