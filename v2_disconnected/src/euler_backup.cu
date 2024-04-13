#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "euler.cuh"
#include "listranking.cuh"

#define DEBUG

EulerianTour::EulerianTour(int nodes, int numComponents) : N(nodes), num_comp(numComponents) {
    edge_count = N - num_comp; 
    edges = edge_count * 2;
    
    mem_alloc();
    mem_init();
}

void EulerianTour::mem_alloc() {
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_child_count, N * sizeof(int)), "Failed to allocate device memory for d_child_count");
    CUDA_CHECK(cudaMalloc(&d_child_num, N * sizeof(int)), "Failed to allocate device memory for d_child_num");
    CUDA_CHECK(cudaMalloc(&starting_index, (N+1) * sizeof(int)), "Failed to allocate memory for starting_index");
    CUDA_CHECK(cudaMalloc(&d_edge_num,  edges * sizeof(int2)), "Failed to allocate memory for d_edge_num");
    CUDA_CHECK(cudaMalloc(&d_successor, edges * sizeof(int)), "Failed to allocate memory for d_successor");
    CUDA_CHECK(cudaMalloc(&d_child_list, edge_count * sizeof(int)), "Failed to allocate memory for d_child_list");
    CUDA_CHECK(cudaMalloc(&d_last_edges, num_comp * sizeof(int)), "Failed to allocate memory for d_last_edges");
    CUDA_CHECK(cudaMalloc(&d_first_edge, num_comp * sizeof(int)), "Failed to allocate memory for d_first_edge");
    CUDA_CHECK(cudaMalloc(&d_rank, edges * sizeof(int)), "Failed to allocate device memory for d_rank");
    CUDA_CHECK(cudaMalloc(&d_new_first, N * sizeof(int)), "Failed to allocate device memory for d_new_first");
    CUDA_CHECK(cudaMalloc(&d_new_last, N * sizeof(int)), "Failed to allocate device memory for d_new_last");
}

void EulerianTour::mem_init() {

    // Initialize device memory
    CUDA_CHECK(cudaMemset(d_child_list, 0, edge_count * sizeof(int)), "Failed to initialize d_child_list");
    CUDA_CHECK(cudaMemset(d_child_count, 0, N * sizeof(int)), "Failed to initialize d_child_count");
    CUDA_CHECK(cudaMemset(d_child_num, 0, N * sizeof(int)), "Failed to initialize d_child_num");
}

EulerianTour::~EulerianTour() {
    // Free allocated device memory
    cudaFree(d_child_count);
    cudaFree(d_child_num);
    cudaFree(starting_index);
    cudaFree(d_edge_num);
    cudaFree(d_successor);
    cudaFree(d_child_list);
    cudaFree(d_last_edges);
    cudaFree(d_first_edge);
    cudaFree(d_rank);
    cudaFree(d_new_first);
    cudaFree(d_new_last);
}

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

__global__
void update_parent(int* d_parents, int* d_roots, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n) {
        #ifdef DEBUG
            printf("d_roots[tid] = %d\n", d_roots[tid]);
        #endif
        d_parents[d_roots[tid]] = -1;
    }
}

__global__ 
void find_degrees(int *d_parent, int *d_child_count, int *d_child_num, int n) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < n && d_parent[idx] != -1) 
    {
       int old_count = atomicAdd(&d_child_count[d_parent[idx]], 1);
        // assign the child a number (old child count of parent)
        d_child_num[idx] = old_count;
    }
}

__global__ 
void populate_child_list_kernel(
    int total_edges, 
    int *prefix_sum, 
    int *d_child_num, 
    int *d_child_list, 
    int *d_parent, 
    int2 *d_edge_num, 
    int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {

        if(d_parent[i] != -1) {

            d_child_list[prefix_sum[d_parent[i]] + d_child_num[i]] = i;
            d_edge_num[prefix_sum[d_parent[i]] + d_child_num[i]] = make_int2(d_parent[i], i);
            d_edge_num[prefix_sum[d_parent[i]] + d_child_num[i] + total_edges] = make_int2(i, d_parent[i]);
        }
    }
}

__device__ __forceinline__ 
int find_edge_num(int u, int v, int* prefix_sum, int* child_num, int* parent, int n_edges) {
    //check if forward edge
    if(parent[v] == u) {
        return prefix_sum[u] + child_num[v];
    } else {
        return prefix_sum[v] + child_num[u] + n_edges;
    }
}

__global__ 
void find_successor_kernel(
    int n_edges, 
    int n_nodes, 
    int *prefix_sum, 
    int *child_num, 
    int *child_list, 
    int *parent, 
    int *successor, 
    int *child_count, 
    int2 *d_edge_num, 
    int *d_last_edge, 
    int *d_first_edge, 
    int *d_rep_map, 
    int total_edges) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < total_edges) {
        
        int u = d_edge_num[i].x;
        int v = d_edge_num[i].y;

        //step 1, check if forward edge, i.e. from parent to child
        if(parent[v] == u) {
            // i] Check if it is the first edge
            if(parent[u] == -1 and !child_num[v]) {
                d_first_edge[d_rep_map[u]] = find_edge_num(u, v, prefix_sum, child_num, parent, n_edges);

                #ifdef DEBUG
                    printf("The first edge : %d ", find_edge_num(u, v, prefix_sum, child_num, parent, n_edges));
                #endif
            }
            // ii] Check if v has any calculate_children

            if(child_count[v] > 0) {
                /* yes then go to the first child of v;
                    new edge will be from v to 0th child of v
                    child_list[prefix_sum[v]] will give me first child of v, 
                    as prefix_sum[v] denotes the starting of child_list of v.  */
            
                successor[i] = find_edge_num(v, child_list[prefix_sum[v]], prefix_sum, child_num, parent, n_edges);
                return;
            }
            else
            {
                //No child, then go back to parent.
                successor[i] = find_edge_num(v, parent[v], prefix_sum, child_num, parent, n_edges);
                return;
            }
        }

        // it is an back-edge
        else 
        {
            //check if it is going back to root  
            if(parent[v] == -1)
            {
                if(child_num[u] == child_count[v] - 1) //checking if it is the last child
                {
                    int val = find_edge_num(u,v, prefix_sum, child_num, parent, n_edges);
                    successor[i] = val;
                    d_last_edge[d_rep_map[v]] = val;
                    //Added this condition check for lone vertices.
                    if(!child_count[v])
                        d_last_edge[d_rep_map[v]] = -1;
                    #ifdef DEBUG
                        printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v]]);
                        // printf("\nSuccessor[%d] = %d", i, val);
                    #endif
                    return;
                }
                else {
                    successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1], prefix_sum, child_num, parent, n_edges);
                    #ifdef DEBUG
                        printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v] + child_num[u] + 1]);
                    #endif
                    return;
                }
            }

            //find child_num of u
            //check if it was the last child of v, if yes then the edge will be back to parent.
            int child_no_u = child_num[u];
            
            if(child_no_u == child_count[v] - 1) {
                //go to the parent of 0;
                successor[i] = find_edge_num(v, parent[v], prefix_sum, child_num, parent, n_edges);
                #ifdef DEBUG
                    printf("\nSuccessor[%d,%d] = %d, %d",u, v, v, parent[v]);
                #endif
            }
            else {
            //It is not the last child
            successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1 ], prefix_sum, child_num, parent, n_edges);
            #ifdef DEBUG
                printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v] + child_num[u] + 1 ]);
            #endif
            }
        }
    }
}

__global__
void update_rank_kernel(
    int *euler_tour_arr, 
    int n_edges, 
    int *rank, 
    int *d_rep_arr, 
    int *d_rep_map, 
    int2 *d_edge_num, 
    int *d_first_edge) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_edges) {

        int k = euler_tour_arr[d_first_edge[d_rep_map[d_rep_arr[d_edge_num[i].x]]]];
        #ifdef DEBUG
            // For debugging: The value of 'k' is computed in a single step below.
            // It's equivalent to the following expanded steps:
            int rep_u = d_rep_arr[d_edge_num[i].x];
            int tree_no = d_rep_map[rep_u];
            int first_edge_tree = d_first_edge[tree_no];
            k = euler_tour_arr[first_edge_tree];
            // printf("\nFor edge %d, the representative is %d and the first edge of the tree is %d and the value to subtract from is %d", i, rep_u, first_edge_tree, k);
        #endif
        rank[i] = k - euler_tour_arr[i];
    }
}

__global__ 
void compute_first_last(int2 *edge_num, int *parent, int *first, int *last, int *euler_tour_arr, int *rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int u = edge_num[i].x;
        atomicMin(&first[u], rank[i]);
        atomicMax(&last[u], rank[i]);
    }
}

__device__ __forceinline__ 
int findedge_num(int u, int v, int* parent, int* prefix_sum, int* child_num, int n_nodes, int n_edges) 
{
    //check if forward edge
    if(parent[v] == u) 
    {
    // printf("\n(%d, %d) <- %d", u, v, prefix_sum[u] + child_num[v]);
        return prefix_sum[u] + child_num[v];
    }
    else {
    // printf("\n(%d, %d) <- %d", u, v, prefix_sum[v] + child_num[u] + + n_nodes - 1);
        return prefix_sum[v] + child_num[u] + n_edges;
    }
}

__global__ 
void compute_first_last_new(
    int* child_count, int* child_num, int* child_list, 
    int* starting_index, int* parent, int* rank, 
    int n, int n_edges, int* new_first_ptr, int* new_last_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) 
        return;

    if (!child_count[i]) {
        // sub_graph_size[i] = 0;
        int k = rank[findedge_num(i, parent[i], parent, starting_index, child_num, n, n_edges)];
        new_first_ptr[i] = new_last_ptr[i] = k;
    } else {

        int node_0 = i;
        int node_1 = child_list[starting_index[i] + 0];
        int node_2 = child_list[starting_index[i] + child_count[i] - 1];
        int edge_1 = findedge_num(node_0, node_1, parent, starting_index, child_num, n, n_edges);
        int edge_2 = findedge_num(node_2, node_0, parent, starting_index, child_num, n, n_edges);
        int edge_3 = findedge_num(node_0, node_2, parent, starting_index, child_num, n, n_edges);

        if(parent[i] != -1) {

            new_first_ptr[i] = rank[edge_1];
            new_last_ptr[i] = rank[findedge_num(node_0, parent[i], parent, starting_index, child_num, n, n_edges)];
        }
        else {
            new_first_ptr[i] = rank[edge_1];
            new_last_ptr[i] = rank[edge_3];
        }

    }
}


/**
 * Performs an Euler tour on a graph.
 *
 * This function conducts an Euler tour on a graph represented by its parent 
 * relationships and returns the first and last occurrence of each node during the tour.
 *
 * @param d_parent: An integer pointer representing the parent of each node in the graph.
 * @param d_roots: An integer pointer marking the root nodes of each connected component in the graph.
 * @param h_first: A host vector that will store the first occurrence of each node during the tour.
 * @param h_last: A host vector that will store the last occurrence of each node during the tour.
 * @param N: The number of nodes in the forest.
 * @param num_comp: The number of trees in the forest.
 */

void euler_tour(
    int* d_parent, int* d_roots, 
    int* d_rep_arr, int* d_rep_map, 
    int nodes, int num_comp, 
    EulerianTour& euler_mag) {
    
    int N = nodes;
    int edge_count  =   N - num_comp;
    int edges       =   edge_count * 2;

    int blockSize   =   1024;
    int gridSize    =   (N + blockSize - 1) / blockSize;

    update_parent<<<gridSize, blockSize>>>(d_parent, d_roots, num_comp);

    // Device pointers
    int* d_child_count  =   euler_mag.d_child_count;
    int* d_child_num    =   euler_mag.d_child_num;
    int* starting_index =   euler_mag.starting_index;
    int* d_successor    =   euler_mag.d_successor;
    int* d_child_list   =   euler_mag.d_child_list;
    int* d_first_edge   =   euler_mag.d_first_edge;
    int* d_last_edges   =   euler_mag.d_last_edges;
    int *d_rank         =   euler_mag.d_rank;
    int *d_new_first    =   euler_mag.d_new_first;
    int *d_new_last     =   euler_mag.d_new_last;
    int2* d_edge_num    =   euler_mag.d_edge_num;

    blockSize = 1024;
    gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel with calculated dimensions
    find_degrees<<<gridSize, blockSize>>>(
        d_parent, 
        d_child_count, 
        d_child_num,
        N);

    cudaDeviceSynchronize();

    /*
    ** Prefix sum is done here:
    ** - The size is one greater than the input size to accommodate the sum that includes all input elements.
    ** - All elements of the starting_index array are initialized to 0, including the first one, as we want the output to start from 0.
    ** - We perform an inclusive scan on the input.
    ** - We use an inclusive scan instead of an exclusive scan because we want to include the last element in the sum.
    ** - The result is stored starting from the second element of the 'd_output' array to ensure it starts from 0.
    */

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_child_count, starting_index + 1, N);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_child_count, starting_index + 1, N);
    cudaDeviceSynchronize();

    int threadsPerBlock = 1024;  // This is a common choice, adjust based on your GPU architecture
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel call
    populate_child_list_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        edge_count,
        starting_index,
        d_child_num,
        d_child_list,
        d_parent,
        d_edge_num,
        N);

    cudaDeviceSynchronize();  // Ensure all operations are finished

    //To be used for list ranking
    blocksPerGrid = (edges + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel call
    find_successor_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        edge_count,
        N,
        starting_index,
        d_child_num,
        d_child_list,
        d_parent,
        d_successor,
        d_child_count,
        d_edge_num,
        d_last_edges,
        d_first_edge,
        d_rep_map,
        edges);

    cudaDeviceSynchronize();  // Ensure all operations are finished

    //apply list ranking on successor to get Euler tour
    //store the tour in a different array 
    int* d_euler_tour_arr = listRanking(d_successor, edges, d_last_edges, num_comp);

    

    //After Eulerian Tour is ready, get the correct ranks
    //Update ranks, then calculate first and last

    //edges is 2 times the original number of edges

    // Device pointer for the rank array
    
    blocksPerGrid = (edges + threadsPerBlock - 1) / threadsPerBlock;
    update_rank_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_euler_tour_arr,
        edges,
        d_rank,
        d_rep_arr,
        d_rep_map,
        d_edge_num,
        d_first_edge
    );

    cudaDeviceSynchronize(); // Ensure all operations are finished

    // Initialise first and last arrays with INT_MAX and 0
    auto start = std::chrono::high_resolution_clock::now();
    
    blocksPerGrid = (edges + threadsPerBlock - 1) / threadsPerBlock;

    printMemoryInfo("After all allocations are done");
    
    auto start_kernel = std::chrono::high_resolution_clock::now();

    compute_first_last_new<<<gridSize, blockSize>>>(
        d_child_count, 
        d_child_num, 
        d_child_list, 
        starting_index, 
        d_parent, 
        d_rank, 
        N, 
        edge_count, 
        d_new_first, 
        d_new_last);

    cudaDeviceSynchronize();
    
    auto end_kernel = std::chrono::high_resolution_clock::now();
    // Compute the difference between the end and start time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count();

    std::cout << "\nKernel without atomic took " << duration << " microseconds." << std::endl;

    #ifdef DEBUG
        std::vector<int> n_first(N);
        std::vector<int> n_last(N);

        // Copy data from device to host
        cudaMemcpy(n_first.data(), d_new_first, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(n_last.data(), d_new_last, N * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Node\tFirst\tLast\n";
        for (int i = 0; i < N; ++i)
        {
            std::cout << "Node " << i << ": " << n_first[i] << "\t" << n_last[i] << "\n";
        }
    #endif
}
