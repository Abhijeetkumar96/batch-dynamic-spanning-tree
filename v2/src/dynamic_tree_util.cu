#include "cuda_utility.cuh"
#include "euler_tour.cuh"
#include "dynamic_tree_util.cuh"
#include "update_ds.cuh"

// #define DEBUG

using namespace cub;

void dynamic_tree_manager::mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list) {

	num_vert = parent.size();
    num_edges = edge_list.size();
    
    size_t size = parent.size() * sizeof(int);
    size_t delete_size = edges_to_delete.size() * sizeof(uint64_t);
    size_t num_edges = edge_list.size() * sizeof(uint64_t);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&d_org_parent, size), "Failed to allocate memory for d_org_parent");
    CUDA_CHECK(cudaMalloc(&d_unique_rep, size), "Failed to allocate memory for d_unique_rep");
    CUDA_CHECK(cudaMalloc(&d_rep_map, size), "Failed to allocate memory for d_rep_map");
    CUDA_CHECK(cudaMalloc(&d_edges_to_delete, delete_size), "Failed to allocate memory for edges to delete");
    
    // d_edge_list is the original edge_list
    CUDA_CHECK(cudaMalloc(&d_edge_list, num_edges), "Failed to allocate memory for input edge list");
    
    // d_updated_edge_list is the new edgelist after deleting the edges
    CUDA_CHECK(cudaMalloc(&d_updated_edge_list, num_edges), "Failed to allocate memory for input edge list");

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), size, cudaMemcpyHostToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_org_parent, d_parent, size,  cudaMemcpyDeviceToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_delete, edges_to_delete.data(), delete_size, cudaMemcpyHostToDevice), "Failed to copy edges to delete to device");
    CUDA_CHECK(cudaMemcpy(d_edge_list, edge_list.data(), num_edges, cudaMemcpyHostToDevice), "Failed to copy edge list to device");
    
    // Create a hash table on the device
    // HashTable = create_hashtable();
}

void dynamic_tree_manager::read_delete_batch(const std::string& delete_filename) {

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
    
    std::cout << "Reading " << n_edges << " edges from the file." << std::endl;

    for (int i = 0; i < n_edges; ++i) {
        inputFile >> u >> v;
        if(u > v) {
            // Ensures u is always less than v for consistent edge representation
            std::swap(u, v);
        }
        edges_to_delete[i] = ((uint64_t)(u) << 32 | v);
    }

    #ifdef DEBUG
        std::cout << "edges_to_delete array:\n";
        host_print(edges_to_delete);
    #endif
}

void dynamic_tree_manager::update_existing_ds() {
	update_edgelist(
        d_parent, num_vert, 
        d_edge_list, d_updated_edge_list, num_edges, 
        d_edges_to_delete, delete_batch_size);

    // now num_edges contains nonTreeEdges - delete_batch count.
}

dynamic_tree_manager::~dynamic_tree_manager() {
    cudaFree(d_parent);
    cudaFree(d_unique_rep);
    cudaFree(d_edges_to_delete);
    cudaFree(d_edge_list);
    // destroy_hashtable(HashTable);
}

// ====[ End of update ds Code ]====