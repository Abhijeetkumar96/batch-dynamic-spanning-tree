#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"
#include "eulerian_tour/connected/euler_tour.cuh"
#include "eulerian_tour/connected/list_ranking.cuh"

// #define DEBUG

namespace sce {

__global__ 
void find_degrees(
    int *d_parent, 
    int *d_child_count, 
    int *d_child_num, 
    int n) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(idx < n && d_parent[idx] != idx) {
        d_child_num[idx] = atomicAdd(&d_child_count[d_parent[idx]], 1);
    }
}

__global__
void populate_child_list_kernel(
    int root, 
    int n_nodes, 
    int *prefix_sum, 
    int *child_num, 
    int *d_child_list, 
    int *parent, 
    int2 *d_edge_num ) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(i < n_nodes && i != root) {

        d_child_list[prefix_sum[parent[i]] + child_num[i]] = i;
        d_edge_num[prefix_sum[parent[i]] + child_num[i]] = make_int2(parent[i],i);
        d_edge_num[prefix_sum[parent[i]] + child_num[i] + n_nodes - 1] = make_int2(i, parent[i]);
    }
}

__device__ __forceinline__ 
int find_edge_num(int u, int v, int* prefix_sum, int* child_num, int* parent, int n_nodes) {
    //check if forward edge
    if(parent[v] == u) {
        return prefix_sum[u] + child_num[v];
    } else {
        return prefix_sum[v] + child_num[u] + n_nodes;
    }
}

__global__
void find_successor_kernel(
    int root, 
    int n_nodes,
    int edges, 
    int* prefix_sum, 
    int* child_num, 
    int* child_list, 
    int* parent, 
    int* successor, 
    int2* d_edge_num, 
    int* child_count,
    int* d_last_edge) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < edges) {
        int u = d_edge_num[i].x;
        int v = d_edge_num[i].y;
        // step 1, check if forward edge, i.e. from parent to child
        if(parent[v] == u) {
            
          // i] Check if v has any calculate_children
            if(child_count[v] > 0) {
                // if yes then go to the first child of v;
                // new edge will be from v to 0th child of v
                // d_child_list[prefix_sum[v]] will give me first child of v, as prefix_sum[v] denotes the starting of d_child_list of v
                successor[i] = find_edge_num(v, child_list[prefix_sum[v]], prefix_sum, child_num, parent, n_nodes);
                return;
            }
            else {
              // No child, then go back to parent.
              successor[i] = find_edge_num(v, parent[v], prefix_sum, child_num, parent, n_nodes);
              return;
            }
        }
    
        // it is an back-edge
        else {
            // check if it is going back to root  
            if(v == root) {
                
                // checking if it is the last child
                if(child_num[u] == child_count[root] - 1) {

                    int val = find_edge_num(root, child_list[prefix_sum[root]], prefix_sum, child_num, parent, n_nodes);
                    successor[i] = val;
                    *d_last_edge = find_edge_num(u, v, prefix_sum, child_num, parent, n_nodes);

                    return;
                }
                else {
                    successor[i] = find_edge_num(v, child_list[prefix_sum[root] + child_num[u] + 1], prefix_sum, child_num, parent, n_nodes);
                    return;
                }
            }
            // find child_num of u
            // check if it was the last child of v, if yes then the edge will be back to parent.
            
            int child_no_u = child_num[u];
            if(child_no_u == child_count[v] - 1) {
                //go to the parent of 0;
                successor[i] = find_edge_num(v, parent[v], prefix_sum, child_num, parent, n_nodes);
            }
            else {
            //It is not the last child
                successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1 ], prefix_sum, child_num, parent, n_nodes);
            }
        }
    }
}

__global__
void update_rank_kernel(int* euler_tour_arr, int n_edges, int* rank) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(i < n_edges) {
        rank[i] = n_edges - 1 - euler_tour_arr[i];
    }
}

__device__ __forceinline__ int findedge_num(int u, int v, int* parent, int* prefix_sum, int* child_num, int n_nodes, int n_edges) 
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
void compute_first_last(
    int* child_count, 
    int* child_num, 
    int* child_list, 
    int* starting_index, 
    int* parent, 
    int* rank, 
    int n, 
    int n_edges, 
    int* new_first_ptr, 
    int* new_last_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) 
        return;

    if (!child_count[i]) {

        int k = rank[findedge_num(i, parent[i], parent, starting_index, child_num, n, n_edges)] - 1;
        new_first_ptr[i] = new_last_ptr[i] = k;
    } else {

        int node_0 = i;
        int node_1 = child_list[starting_index[i] + 0];
        int node_2 = child_list[starting_index[i] + child_count[i] - 1];
        int edge_1 = findedge_num(node_0, node_1, parent, starting_index, child_num, n, n_edges);
        int edge_2 = findedge_num(node_2, node_0, parent, starting_index, child_num, n, n_edges);
        int edge_3 = findedge_num(node_0, node_2, parent, starting_index, child_num, n, n_edges);

        if(parent[i] != i) {
            new_first_ptr[i] = rank[edge_1] - 1;
            new_last_ptr[i] = rank[findedge_num(node_0, parent[i], parent, starting_index, child_num, n, n_edges)] - 1;
        } else {
            new_first_ptr[i] = rank[edge_1] - 1;
            new_last_ptr[i] = rank[edge_3] - 1;
        }
    }
}


__global__
void update_root_last(
    int* new_last_ptr, 
    int* child_count, 
    int* child_list, 
    int* starting_index,
    int root) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i == 0) {
        int last_child_root = child_list[starting_index[root] + child_count[root] - 1];
        new_last_ptr[root] = new_last_ptr[last_child_root] + 1;
    }
}

/**
 * Function: parallelSubgraphComputation()
 *
 * Description:
 *   - This function conducts an Euler tour on a graph represented by its parent 
 *     relationships and returns the child_count of each node.
 *
 * Parameters:
 *   - @param d_parent: An integer pointer representing the parent of each node in the graph. 
 *                     Essential for actual computation.
 *   - @param root: The root node of the tree.
 *   - @param N: The total number of nodes in the forest.
 *   
 *   - @param d_child_count: An integer pointer for storing the count of children for each node.
 *   - @param d_child_num: An integer pointer storing a unique number for each child in the graph.
 *   - @param starting_index: An integer pointer representing the starting index for each node 
 *                            in the csr_graph.
 *
 * Notes:
 *   - The function assumes d_parent, root, and N are primary inputs for computation.
 *   - The arrays d_child_count, d_child_num, and starting_index serve as intermediate storage 
 *     for results and necessary computations.
 */
 void cal_first_last(int root, int* d_parent, EulerianTour* eulerTour) {
    
    // std::cout << "root: " << root << std::endl;

    const int numNodes      =   eulerTour->N;
    const int edge_count    =   numNodes - 1;
    const int edges         =   edge_count * 2;

    int* d_child_count      =   eulerTour->d_child_count; 
    int* d_child_num        =   eulerTour->d_child_num;
    int* d_child_list       =   eulerTour->d_child_list;
    int2* d_edge_num         =   eulerTour->d_edge_num;
    int* starting_index     =   eulerTour->starting_index;
    int* new_first          =   eulerTour->new_first;
    int* new_last           =   eulerTour->new_last;
    int* d_successor        =   eulerTour->successor;
    int* d_last_edge        =   eulerTour->d_last_edge;
    int* d_euler_tour_arr   =   eulerTour->d_euler_tour_arr;

    int numThreads = 1024;
    int numBlocks = (numNodes + numThreads - 1) / numThreads;

    // Launch kernel with calculated dimensions
    find_degrees<<<numBlocks, numThreads>>>(
        d_parent, 
        d_child_count, 
        d_child_num, 
        numNodes);

    CUDA_CHECK(cudaGetLastError(), "Error after find_degrees kernel launch");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to syncronize after find_degrees");

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
    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_child_count, starting_index + 1, numNodes);
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temp storage");
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_child_count, starting_index + 1, numNodes);
    // thrust::inclusive_scan(thrust::device, d_child_count, d_child_count + numNodes, starting_index + 1);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    numBlocks = (numNodes + numThreads - 1) / numThreads;

    populate_child_list_kernel<<<numBlocks, numThreads>>>(
        root, 
        numNodes, 
        starting_index, 
        d_child_num, 
        d_child_list, 
        d_parent, 
        d_edge_num);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    #ifdef DEBUG
        int2* h_edge_num = new int2[edges]; // 'edges' is the number of elements in 'd_edge_num'
        cudaMemcpy(h_edge_num, d_edge_num, edges * sizeof(int2), cudaMemcpyDeviceToHost);
        std::cout << "\nPrinting edge numbers: \n";
        for(int i = 0; i < edges; ++i) {
            std::cout << i << " : (" << h_edge_num[i].x << ", " << h_edge_num[i].y << ")" << std::endl;
        }

        delete[] h_edge_num;
    #endif

    numBlocks = (edges + numThreads - 1) / numThreads;

    find_successor_kernel<<<numBlocks, numThreads>>>(
        root, 
        edge_count, 
        edges, 
        starting_index,
        d_child_num, 
        d_child_list,
        d_parent, 
        d_successor,
        d_edge_num,
        d_child_count,
        d_last_edge);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    #ifdef DEBUG
        //Copy back the data
        std::vector<int> h_successor(edges);
        cudaMemcpy(h_successor.data(), d_successor, edges*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout<<"\nPrinting successor array : \n";
        for(int i = 0; i<h_successor.size(); ++i)
        {
          std::cout<<"successor["<<i<<"] = "<<h_successor[i]<<"\n";
        }

        std::cout<<"\nPrinting last_edge array : \n";
        std::cout << *d_last_edge << std::endl;
        
    #endif

    //apply list ranking on successor to get Euler tour
    cuda_list_rank(edges, *d_last_edge, d_successor, d_euler_tour_arr, eulerTour->getListRanking());

    #ifdef DEBUG
        std::vector<int> h_euler_tour_arr(edges);
        cudaMemcpy(h_euler_tour_arr.data(), d_euler_tour_arr, edges*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout <<"Euler tour array after applying listranking:\n";
        int jj = 0;
        for(auto i : h_euler_tour_arr)
            std::cout << "arr["<< jj++ <<"] : " << i << std::endl;
        std::cout << std::endl;
    #endif


    //After Eulerian Tour is ready, get the correct ranks
    //Update ranks, then calculate first and last

    //edges is 2 times the original number of edges

    // numBlocks = (edges + numThreads - 1) / numThreads;

    // update_rank_kernel<<<numBlocks, numThreads>>>(d_euler_tour_arr, edges, rank);
    // cudaDeviceSynchronize();

    numBlocks = (numNodes + numThreads - 1) / numThreads;

    compute_first_last<<<numBlocks, numThreads>>>(
        d_child_count, 
        d_child_num, 
        d_child_list, 
        starting_index, 
        d_parent, 
        d_euler_tour_arr, 
        numNodes, 
        edge_count, 
        new_first, 
        new_last);

    update_root_last<<<1,1>>>(
        new_last, 
        d_child_count, 
        d_child_list, 
        starting_index, 
        root);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    // bool g_verbose = true;
    
    if(g_verbose) {
        int* h_first = new int[numNodes];
        int* h_last = new int[numNodes];
        
        // Step 2: Copy data from device to host
        cudaMemcpy(h_first, new_first, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_last, new_last, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Node\tFirst\tLast\n";
        
        for (int i = 0; i < numNodes; ++i) {
            std::cout << "Node " << i << ": " << h_first[i] << "\t" << h_last[i] << "\n";
        }    
    }
}
} // namespace sce