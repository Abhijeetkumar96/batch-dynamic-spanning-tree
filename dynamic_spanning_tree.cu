#include "cuda_utility.cuh"
#include "euler_tour.cuh"
#include "dynamic_tree.cuh"

__device__ __forceinline__
long binary_search(uint64_t* array, long num_elements, uint64_t key) {
    long left = 0;
    long right = num_elements - 1;
    while (left <= right) {
        long mid = left + (right - left) / 2;
        if (array[mid] == key) {
            return mid; // Key found
        }
        if (array[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // Key not found
}

__global__
void delete_edges_kernel(
    int* d_parent,          // size <- numNodes
    uint64_t* d_edge_list,  // size <- numEdges
    long num_edges,         
    uint64_t* d_edges_to_delete, // size <- delete_batch_size
    int delete_batch_size, 
    unsigned char* flags)       // size <- numEdges
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < delete_batch_size) {
        uint32_t u, v;
        uint64_t t = d_edges_to_delete[tid];
        uint32_t u = (uint32_t)(t >> 32);
        uint32_t v = (uint32_t)(t & 0xFFFFFFFF); 

        // delete tree edges
        if(u == d_parent[v]) {
            d_parent[v] = v;
        }
        else if(v == parent[u]) {
            parent[u] = u;
        }

        else {
            // t is the key, to be searched in the d_edge_list array
            long pos = binary_search(d_edge_list, num_edges, t);
            if(pos != -1) {
                d_flag[pos] = 0;
            }
        }
    }
}

void sort_array_uint64_t(uint64_t* d_data, long num_items) {
    // Allocate temporary storage for sorting
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto start = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items);
    cudaDeviceSynchronize();
}

void select_flagged(uint64_t* d_in, uint64_t* d_out, unsigned char* d_flags, long& num_items) {
    // Allocate device output array and num selected
    long *d_num_selected_out   = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(uint64_t) * num_items), "Failed to allocate memory for d_out");
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(long)), "Failed to allocate memory for d_num_selected_out");

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after cub::Flagged");

    CUDA_CHECK(cudaMemcpyAsync(&num_items, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost),"Failed to copy back d_num_selected_out");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to copy back num_items");
}

void update_existing_ds(
    int* d_parent, int* d_rep, int num_vert, 
    uint64_t* d_edge_list, long num_edges, 
    uint64_t* d_edges_to_delete, int delete_size) {

    // sort the input edges
    sort_array_uint64_t(d_edge_list, num_edges);
    
    // init d_flag with true values
    unsigned char   *d_flags = NULL;
    std::vector<unsigned char> h_flags(num_edges, 1);
    CUDA_CHECK(cudaMalloc((void**)&d_flags, sizeof(unsigned char) * num_edges), 
        "Failed to allocate memory for d_flags");

    delete_edges_kernel<<<<<<numThreads, numBlocks>>>(
        d_parent, 
        d_edge_list, 
        num_edges, 
        d_edges_to_delete, 
        delete_size, 
        d_flags
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to delete edges");

    CUDA_CHECK(
        cudaMemcpyAsync(
            d_rep, 
            d_parent, 
            num_vert * sizeof(int), 
            cudaMemcpyDeviceToDevice,
            ), 
        "Failed to copy parent from device to device"
    );

    uint64_t* d_updated_ed_list = nullptr;

    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, sizeof(unsigned char) * num_items, cudaMemcpyHostToDevice);
    
    // now delete the edges from the parent array
    select_flagged(d_edge_list, d_updated_ed_list, d_flags, num_edges);
}
    
void dynamic_tree_manager::mem_alloc(const std::vector<int>& parent, const std::vector<uint64_t>& edge_list) {

	num_vert = parent.size();
    num_edges = edge_list.size();
    
    size_t size = parent.size() * sizeof(int);
    size_t delete_size = edges_to_delete.size() * sizeof(uint64_t);
    size_t num_edges = edge_list.size() * sizeof(uint64_t);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_parent, size), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc(&d_rep, size), "Failed to allocate memory for d_rep");
    CUDA_CHECK(cudaMalloc(&d_unique_rep, size), "Failed to allocate memory for d_unique_rep");
    CUDA_CHECK(cudaMalloc(&d_edges_to_delete, delete_size), "Failed to allocate memory for edges to delete");
    CUDA_CHECK(cudaMalloc(&d_edge_list, num_edges), "Failed to allocate memory for input edge list");

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), size, cudaMemcpyHostToDevice), "Failed to copy d_parent to device");
    CUDA_CHECK(cudaMemcpy(d_edges_to_delete, edges_to_delete.data(), delete_size, cudaMemcpyHostToDevice), "Failed to copy edges to delete to device");
    CUDA_CHECK(cudaMemcpy(d_edge_list, edge_list.data(), num_edges, cudaMemcpyHostToDevice), "Failed to copy edge list to device");
    
    // Create a hash table on the device
    pHashTable = create_hashtable();
}

dynamic_tree_manager::~dynamic_tree_manager() {
    cudaFree(d_parent);
    cudaFree(d_rep);
    cudaFree(d_unique_rep);
    cudaFree(d_edges_to_delete);
    cudaFree(d_edge_list);
    destroy_hashtable(pHashTable);
}

keyValues* create_hashtable() {
    keyValues* hashtable;
    CUDA_CHECK(cudaMalloc(&hashtable, sizeof(keyValues) * kHashTableCapacity), "Failed to allocate hashtable");
    CUDA_CHECK(cudaMemset(hashtable, 0xff, sizeof(keyValues) * kHashTableCapacity), "Failed to initialize hashtable");
    return hashtable;
}

void destroy_hashtable(keyValues* pHashTable) {
    CUDA_CHECK(cudaFree(pHashTable), "Failed to free hashtable");
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
}

void dynamic_tree_manager::update_existing_ds() {
	update_existing_ds(d_parent, d_rep, num_vert, d_edge_list, num_edges, d_edges_to_delete, delete_batch_size);
}

void repair_spanning_tree() {
	
	cal_first_last(roots[0], d_parent, euler_tour);

}



// ====[ End of update ds Code ]====