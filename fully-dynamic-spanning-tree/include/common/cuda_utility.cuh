/******************************************************************************
* Functionality: GPU related Utility manager
 ******************************************************************************/

#ifndef CUDA_UTILITY_H
#define CUDA_UTILITY_H

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
extern bool checker;
extern bool g_verbose;
extern long maxThreadsPerBlock;

inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// #define DEBUG

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

// Function to print available and total memory
inline void printMemoryInfo(const std::string& message) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << message << ": Used GPU memory: " 
        << used_db / (1024.0 * 1024.0) << " MB, Free GPU Memory: " 
        << free_db / (1024.0 * 1024.0) << " MB, Total GPU Memory: " 
        << total_db / (1024.0 * 1024.0) << " MB" << std::endl;
}

inline void cuda_init(int device) {
    // Set the CUDA device
    CUDA_CHECK(cudaSetDevice(device), "Unable to set device ");
    cudaGetDevice(&device); // Get current device

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    cudaFree(0);

    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    // int maxBlocksPerGrid = deviceProp.maxGridSize[0]; // Maximum number of blocks per grid
    // std::cout << "maxBlocksPerGrid: " << maxBlocksPerGrid << std::endl;
}

template <typename T>
__global__ 
void print_device_array_kernel(T* array, long size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { 
        for (int i = 0; i < size; ++i) {
            printf("Array[%d] = %d\n", i, array[i]);
        }
    }
}

template <typename T>
__global__ 
void print_device_edge_list_kernel(T* edgeList, long numEdges) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { 
        for (int i = 0; i < numEdges; ++i) {
            int u = edgeList[i] >> 32;  // Extract higher 32 bits
            int v = edgeList[i] & 0xFFFFFFFF; // Extract lower 32 bits
            printf("Edge[%d] = (%d, %d)\n", i, u, v);
        }
    }
}



void pointer_jumping(int* d_next, int n);

template <typename T>
inline void host_print(const std::vector<T> arr) {
    for(const auto &i : arr)
        std::cout << i <<" ";
    std::cout << std::endl;
}

template <typename T>
inline void print_device_array(const T* arr, long size) {
    print_device_array_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_array_kernel");
}

template <typename T>
inline void print_device_edge_list(const T* arr, long size) {
    print_device_edge_list_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_edge_list_kernel");
}

inline void DisplayDeviceEdgeList(const int *device_u, const int *device_v, size_t num_edges) {
    std::cout << "\n" << "Edge List:" << "\n";
    int *host_u = new int[num_edges];
    int *host_v = new int[num_edges];
    cudaMemcpy(host_u, device_u, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_v, device_v, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < num_edges; ++i) {
        std::cout << " Edge[" << i << "]: (" << host_u[i] << ", " << host_v[i] << ")" << "\n";
    }
    delete[] host_u;
    delete[] host_v;
}

// Template function to compare two arrays
template<typename T>
bool compare_arrays(const T* array1, const T* array2, T size) {
    
    for (size_t i = 0; i < size; ++i) {
        if (array1[i] != array2[i]) {
            return false; // Elements at the same position are not equal
        }
    }
    
    return true; // All elements are equal
}

void remove_self_loops_duplicates(
    int*&           d_keys,               // Input keys (edges' first vertices)
    int*&           d_values,             // Input values (edges' second vertices)
    int             num_items,
    uint64_t*&      d_merged,             // Intermediate storage for merged (zipped) keys and values
    unsigned char*& d_flags,              // Flags used to mark items for removal
    int*            d_num_selected_out,   // Output: number of items selected (non-duplicates, non-self-loops)
    int*&           d_keys_out,           // Output keys (processed edges' first vertices)
    int*&           d_values_out);         // Output values (processed edges' second vertices)

void select_flagged(int* d_in, int* d_out, unsigned char* d_flags, int& num_items);
void select_flagged(uint64_t* d_in, uint64_t* d_out, unsigned char* d_flags, long& num_items);

// Function to extract the filename with extension from a filesystem path
inline std::string extract_filename(const std::filesystem::path& filePath) {
    return filePath.filename().string();
}

/*
    Scenario 1:
    - num_items             <-- [8]
    - d_in                  <-- [0, 2, 2, 9, 5, 5, 5, 8]
    - d_out                 <-- [0, 2, 9, 5, 8]
    - d_num_selected_out    <-- [5]    

    Scenario 2:
    - d_in                  <-- [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    - d_out                 <-- [0, 1, 0] 

    So we need to have the array sorted
    ** all indices marked as 1 get to stay
*/
// void find_unique(
//     int* d_in, 
//     int* d_out,
//     int num_items,
//     int& h_num_selected_out);

// template <typename T1, typename T2>
// inline void find_unique(
//     T1* d_in, 
//     T1* d_out,
//     T2 num_items,
//     T2& h_num_selected_out,
//     bool print = false) {
    
//     size_t temp_storage_bytes   =   0;
//     void* d_temp_storage        =   nullptr;
//     T2* d_num_selected_out      =   nullptr;

//     // Allocate device memory for storing the number of unique elements selected
//     CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(T2)), "Failed to allocate memory for d_num_selected_out");
    
//     // Query temporary storage requirements for sorting
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);
//     // Allocate temporary storage
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate memory for d_num_selected_out");
//     // Run sorting operation
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_in, num_items);

//     CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free d_temp_storage");

//     print = false;
    
//     // print d_in values 
//     if(print) {
//         int* h_in = new int[num_items];
//         CUDA_CHECK(cudaMemcpy(h_in, d_in, num_items * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back sorted results");
//         std::cout << "sorted array output:\n";
//         for(int i = 0; i < num_items; ++i) {
//             std::cout << h_in[i] << "\n";
//         }
//         std::cout << std::endl;

//         if (std::is_sorted(h_in, h_in + num_items)) {
//             std::cout << "Sorted in the range." << std::endl;
//         } else {
//             std::cout << "Not Sorted in the range. " << std::endl;
//         }
//     }

//     temp_storage_bytes = 0;
//     d_temp_storage = nullptr;
//     cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);
//     // Allocate temporary storage
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate memory for d_num_selected_out");
//     // Run unique selection operation
//     cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items);

//     // Copy the number of unique elements selected back to host
//     CUDA_CHECK(cudaMemcpy(&h_num_selected_out, d_num_selected_out, sizeof(T2), cudaMemcpyDeviceToHost), "Failed to allocate memory for d_num_selected_out");

//     // Cleanup
//     CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free d_temp_storage");
//     CUDA_CHECK(cudaFree(d_num_selected_out), "Failed to free d_temp_storage");
// }

template <typename T>
inline void DisplayResults(T* arr, int num_items) {
    for(int i = 0; i < num_items; ++i) {
        printf("%llu ", (unsigned long long)arr[i]);
    }
    printf("\n");
}

inline void DisplayDeviceUint64Array(uint64_t* d_arr, unsigned char* d_flags, int num_items) {
    // Allocate host memory for the copy
    uint64_t* h_arr = new uint64_t[num_items];
    
    // Copy data from device to host
    cudaMemcpy(h_arr, d_arr, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);
    
    unsigned char* flag_arr = new unsigned char[num_items];
    cudaMemcpy(flag_arr, d_flags, sizeof(unsigned char) * num_items, cudaMemcpyDeviceToHost);
    
    std::cout << "Device h_in Array: \n";
    for(int i = 0; i < num_items; ++i) {
        std::cout << h_arr[i] << " <-- " << static_cast<int>(flag_arr[i]) << "\n";
    }
    std::cout << std::endl;
    
    // Cleanup host memory
    delete[] h_arr;
    delete[] flag_arr;
}

inline void DisplayDeviceintArray(int* d_arr, unsigned char* d_flags, int num_items) {
    // Allocate host memory for the copy
    int* h_arr = new int[num_items];
    
    // Copy data from device to host
    cudaMemcpy(h_arr, d_arr, sizeof(int) * num_items, cudaMemcpyDeviceToHost);
    
    unsigned char* flag_arr = new unsigned char[num_items];
    cudaMemcpy(flag_arr, d_flags, sizeof(unsigned char) * num_items, cudaMemcpyDeviceToHost);
    
    std::cout << "Device h_in Array: \n";
    for(int i = 0; i < num_items; ++i) {
        std::cout << h_arr[i] << " <-- " << static_cast<int>(flag_arr[i]) << "\n";
    }
    std::cout << std::endl;
    
    // Cleanup host memory
    delete[] h_arr;
    delete[] flag_arr;
}

inline void DisplayDeviceUCharArray(unsigned char* d_arr, int num_items) {
    // Allocate host memory for the copy
    unsigned char* h_arr = new unsigned char[num_items];
    
    // Copy data from device to host
    cudaMemcpy(h_arr, d_arr, sizeof(unsigned char) * num_items, cudaMemcpyDeviceToHost);
    
    std::cout << "Device h_flags Array: ";
    for(int i = 0; i < num_items; ++i) {
        std::cout << static_cast<int>(h_arr[i]) << " "; // Cast to int for clearer output
    }
    std::cout << std::endl;
    
    // Cleanup host memory
    delete[] h_arr;
}

inline int numberOfComponents(std::vector<int> &par)
{
    std::map<int,int> components;
    for(auto i : par)
    {
        components[i]++;
    }
    return components.size();
}

inline int markedComponents(std::vector<int> &marked)
{
    int count = 0;
    for(auto i : marked)
    {
        count += (i != -1);
    }
    return count;
}

inline int rootedComponents(std::vector<int> &par)
{
    int count = 0;
    for (int i = 0; i < (int)par.size(); i++)
    {
        count += (i == par[i]);
    }
    return count;
    
}

inline bool findEdge(const std::vector<std::pair<int, int>> &edge_stream, std::pair<int, int> target)
{
    return std::find(edge_stream.begin(), edge_stream.end(), target) != edge_stream.end();
}

inline std::vector<int> computeDepths(const std::vector<int> &parent)
{
    int n = parent.size();
    std::vector<int> depth(n, -1);

    for (int i = 0; i < n; ++i)
    {
        int current_depth = 0;
        int j = i;

        // Follow parent pointers to compute depth
        while (depth[j] == -1 && parent[j] != j)
        {
            j = parent[j];
            current_depth++;
        }

        // Set the depth
        depth[i] = current_depth + depth[j];
    }

    return depth;
}

inline bool validateRST(const std::vector<int> &parent)
{
    int n = parent.size();
    int cnt_root = 0,root = -1;
    std::vector<int> root_nodes;
    for (int i = 0; i < n; i++)
    {
        if(parent[i] == i)
        {
            cnt_root++;
            root = i;
            root_nodes.push_back(i);
        }
    }

    if(cnt_root == 0)
    {
        std::cout<<"Wrong : No root exist"<<std::endl;
        return false;       
    }

    if(cnt_root != 1)
    {
        std::cout<<"Wrong : Multiple root nodes exist"<<std::endl;
        for(auto i : root_nodes)
        {
            std::cout<<i<<' ';
        }
        std::cout<<std::endl;
        return false;
    }

    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; i++)
    {
        if(i!=root)
            adj[parent[i]].push_back(i);
        // adj[i].push_back(parent[i]);
    }

    std::vector<int> vis(n,0);
    
    std::function<bool(int)> dfs = [&](int src)
    {
        vis[src] = 1;
        for(auto child : adj[src])
        {
            if(vis[child] == 0)
            {
                if(dfs(child))
                {
                    return true;
                }
            }
            else if(vis[child] == 1)
            {
                return true;
            }
        }
        vis[src] = 2;
        return false;
    };
    
    if(dfs(root))
    {
        std::cout<<"Wrong : Tree has cycles - node["<<root<<"]"<<std::endl;
        return false;
    }

    // for(auto i : adj)
    // {
    //  for(auto j : i)
    //      std::cout<<j<<' ';
    //  std::cout<<std::endl;
    // }


    for(int i=0;i<n;i++)
    {
        // for(auto i : vis) std::cout<<i<<' ';
        // std::cout<<std::endl;
        if(vis[i] != 2)
        {   
            std::cout<<"Wrong : Tree has cycles - node["<<i<<"]"<<std::endl;
            return false;
        }
    }

    return true;
}

inline int treeDepth(const std::vector<int> &parent)
{
    std::vector<int> depths = computeDepths(parent);
    return *std::max_element(depths.begin(), depths.end());
}

bool is_tree_or_forest(const int* d_parent, const int num_vert, int& root);

#endif // CUDA_UTILITY_H