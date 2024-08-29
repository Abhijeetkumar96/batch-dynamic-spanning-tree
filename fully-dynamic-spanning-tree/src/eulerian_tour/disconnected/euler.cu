#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "common/cuda_utility.cuh"
#include "eulerian_tour/disconnected/euler_tour.cuh"
#include "eulerian_tour/disconnected/list_ranking.cuh"

// #define DEBUG

namespace mce {

__global__
void update_parent(int* d_parents, int* d_roots, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n) {
        #ifdef DEBUG
            printf("d_roots[tid] = %d\n", d_roots[tid]);
        #endif
        d_parents[d_roots[tid]] = -1;
    }
}

__global__ 
void find_degrees(int *d_parent, int *d_child_count, int *d_child_num, int n) {
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

            // printf("Index: %d, Parent: %d, ChildNum: %d, PrefixSumParent: %d, EdgeIndex: %d\n",
            //    i, d_parent[i], d_child_num[i], prefix_sum[d_parent[i]], prefix_sum[d_parent[i]] + d_child_num[i]);

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
                    // successor[i] = val;
                    successor[i] = -1;
                    return;
                }
                else {
                    successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1], prefix_sum, child_num, parent, n_edges);
                    return;
                }
            }

            //find child_num of u
            //check if it was the last child of v, if yes then the edge will be back to parent.
            int child_no_u = child_num[u];
            
            if(child_no_u == child_count[v] - 1) {
                //go to the parent of 0;
                successor[i] = find_edge_num(v, parent[v], prefix_sum, child_num, parent, n_edges);
            }
            else {
            //It is not the last child
            successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1 ], prefix_sum, child_num, parent, n_edges);
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

__device__ __forceinline__ 
int findedge_num(int u, int v, int* parent, int* prefix_sum, int* child_num, int n_edges) {
    //check if forward edge
    if(parent[v] == u) {
        return prefix_sum[u] + child_num[v];
    } else {
        return prefix_sum[v] + child_num[u] + n_edges;
    }
}

__global__ 
void compute_first_last_new(
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

    // if no children or 0 child count, then it is either a leaf node or a lone vertex
    if (!child_count[i]) {
        // sub_graph_size[i] = 0;
        // Zero children & parent[i] != -1 means it is a leaf node
        if(parent[i] != -1) {
            int edge_num = findedge_num(i, parent[i], parent, starting_index, child_num, n_edges);
            // printf("\nFor lone vertex i: %d, parent[i]: %d, edge_num: %d", i, parent[i], edge_num);
            
            int k = rank[edge_num];
            new_first_ptr[i] = k;
            new_last_ptr[i] = k;
        } 
        // else it is a lone vertex (zero children & parent == -1)
        else { 
            new_first_ptr[i] = 0;
            new_last_ptr[i] = 1;
        }
    } else {

        int node_0 = i;
        int node_1 = child_list[starting_index[i]];
        int node_2 = child_list[starting_index[i] + child_count[i] - 1];

        int edge_1 = findedge_num(node_0, node_1, parent, starting_index, child_num, n_edges);
        int edge_2 = findedge_num(node_2, node_0, parent, starting_index, child_num, n_edges);
        int edge_3 = findedge_num(node_0, node_2, parent, starting_index, child_num, n_edges);

        // printf("\nnode_0: %d, node_1: %d, edge_num: %d", node_0, node_1, edge_1);
        // printf("\nnode_2: %d, node_0: %d, edge_num: %d", node_2, node_0, edge_2);
        // printf("\nnode_0: %d, node_2: %d, edge_num: %d", node_0, node_2, edge_3);

        // if(edge_1 > 2* n_edges || edge_2 > 2* n_edges || edge_3 > 2*n_edges)
        //     printf("\nedge_num out of bound at line 232, edge_num_1: %d, edge_num_2: %d, edge_num_3: %d", edge_1, edge_2, edge_3);

        if(parent[i] != -1) {
            new_first_ptr[i] = rank[edge_1];
            int edge_4 = findedge_num(node_0, parent[i], parent, starting_index, child_num, n_edges);
            
            // printf("\nnode_0: %d, parent[i]: %d, edge_num: %d", node_0, parent[i], edge_4);
            
            // if(edge_4 > 2*n_edges)
            //     printf("\nedge_num out of bound at line 239, edge_num: %d", edge_4);

            new_last_ptr[i] = rank[edge_4];
        }
        else {
            new_first_ptr[i] = rank[edge_1];
            new_last_ptr[i] = rank[edge_3];
        }
    }
}

__global__
void update_root_last(
    int* new_last_ptr, 
    int* child_count, 
    int* child_list, 
    int* starting_index,
    int* d_roots,
    int num_comp) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < num_comp) {
        int root = d_roots[i];
        if(!child_count[root])
            return;
        int last_child_root = child_list[starting_index[root] + child_count[root] - 1];
        new_last_ptr[root] = new_last_ptr[last_child_root] + 1;
    }
}

__global__ 
void printInt2Array(int2* d_edge_num, int num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) {
        for(int i = 0; i < num_edges; ++i)
            printf("Edge %d: (%d, %d)\n", i, d_edge_num[i].x, d_edge_num[i].y);
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

void cal_first_last(
    int* d_parent, int* d_roots, 
    int* d_rep_arr, int* d_rep_map, 
    int nodes, int num_comp, 
    EulerianTour* euler_mag) {
    
    int N = nodes;
    int edge_count  =   N - num_comp;
    int edges       =   edge_count * 2;

    int blockSize   =   1024;
    int num_blocks_vert    =   (N + blockSize - 1) / blockSize;

    g_verbose = false;

    if(g_verbose) {
        std::cout << "\nN: " << N << ", edge_count: " << edge_count << " & edges: " << edges << "\n";
        std::cout << "Printing from cal_first_last:\n";
        std::cout << "d_parent array:\n";
        print_device_array(d_parent, nodes);
        std::cout << "d_roots array:\n";
        print_device_array(d_roots, nodes);
        std::cout << "d_rep_arr array:\n";
        print_device_array(d_rep_arr, nodes);
        std::cout << "d_rep_map array:\n";
        print_device_array(d_rep_map, nodes);    
    }
    
    // g_verbose = false;
    auto start = std::chrono::high_resolution_clock::now();
    update_parent<<<num_blocks_vert, blockSize>>>(d_parent, d_roots, num_comp);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_parent kernel");

    // std::cout << "update_parent kernel over.\n";

    if(g_verbose) {
        std::cout << "updated_parent array:\n";
        print_device_array(d_parent, nodes);
    }

    // Device pointers
    int* d_child_count          =       euler_mag->d_child_count;
    int* d_child_num            =       euler_mag->d_child_num;
    int* starting_index         =       euler_mag->starting_index;
    int* d_successor            =       euler_mag->d_successor;
    int* d_child_list           =       euler_mag->d_child_list;
    int* d_first_edge           =       euler_mag->d_first_edge;
    int *d_rank                 =       euler_mag->d_rank;
    int *d_new_first            =       euler_mag->d_new_first;
    int *d_new_last             =       euler_mag->d_new_last;
    // replace d_euler_tour_arr with d_euler_path
    int *d_euler_tour_arr       =       euler_mag->d_euler_tour_arr;
    int *notAllDone             =       euler_mag->notAllDone;
    int *devNotAllDone          =       euler_mag->devNotAllDone;
    int2* d_edge_num            =       euler_mag->d_edge_num;
    unsigned long long *devRankNext =   euler_mag->devRankNext;

    // Launch kernel with calculated dimensions
    find_degrees<<<num_blocks_vert, blockSize>>>(
        d_parent, 
        d_child_count, 
        d_child_num,
        N);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize find_degrees kernel");
    // std::cout << "find_degrees kernel over.\n";

    #ifdef DEBUG
        std::cout << "Child count array:\n";
        print_device_array(d_child_count, N);
        std::cout << "Child num array:\n";
        print_device_array(d_child_num, N);
    #endif

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
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate d_temp_storage");
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_child_count, starting_index + 1, N);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cub::inc_sum");

    // std::cout << "inc sum over.\n";

    if(g_verbose) {
        std::cout << "Prefix Sum array:\n";
        print_device_array(starting_index, N + 1);
    }

    int threadsPerBlock = 1024;
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

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize populate_child_list kernel");

    // std::cout << "populate_child_list kernel over.\n";

    // g_verbose = true;
    if(g_verbose) {
        std::cout << "d_edge_num array:\n";
        printInt2Array<<<1,1>>>(d_edge_num, edges);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    }
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
        d_first_edge,
        d_rep_map,
        edges);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize find_successor kernel");
    // std::cout << "find_successor kernel over.\n";
    #ifdef DEBUG
        // print successor array
        std::vector<int> h_successor(edges);
            cudaMemcpy(h_successor.data(), d_successor, edges * sizeof(int), cudaMemcpyDeviceToHost);
            for(int i = 0; i < edges; ++i) {
                std::cout << "\nsuccessor[" << i << "]= " << h_successor[i];
            }
        std::cout << std::endl;
    #endif

    //apply list ranking on successor to get Euler tour
    CudaSimpleListRank(d_successor, d_euler_tour_arr, edges, notAllDone, devNotAllDone, devRankNext);
    // std::cout << "list ranking kernel over.\n";

    #ifdef DEBUG
        // print successor array
        std::vector<int> h_eulerian_tour(edges);
            cudaMemcpy(h_eulerian_tour.data(), d_euler_tour_arr, edges * sizeof(int), cudaMemcpyDeviceToHost);
            for(int i = 0; i < edges; ++i) {
                std::cout << "\neulerian_tour[" << i << "]= " << h_eulerian_tour[i];
            }
        std::cout << std::endl;
    #endif

    //After Eulerian Tour is ready, get the correct ranks
    //Update ranks, then calculate first and last

    //edges is 2 times the original number of edges
    
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

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_rank kernel");

    // std::cout << "update_rank kernel over.\n";

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

    // std::cout << "My Eulerian Tour: " << duration << " ms.\n";

    if(g_verbose) {
        std::cout << "d_rank array:\n";
        print_device_array(d_rank, edges);
    }

    compute_first_last_new<<<num_blocks_vert, blockSize>>>(
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
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize compute_first_last kernel");
    // std::cout << "compute_first_last_new kernel over.\n";

    blocksPerGrid = (num_comp + threadsPerBlock - 1) / threadsPerBlock;
    
    update_root_last<<<blocksPerGrid, blockSize>>>(
        d_new_last, 
        d_child_count, 
        d_child_list, 
        starting_index, 
        d_roots,
        num_comp);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_root_last kernel");
    
    #ifdef DEBUG
        std::vector<int> n_first(N);
        std::vector<int> n_last(N);

        // Copy data from device to host
        CUDA_CHECK(cudaMemcpy(n_first.data(), d_new_first, N * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy back d_new_first");
        CUDA_CHECK(cudaMemcpy(n_last.data(), d_new_last, N * sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy back d_new_last");

        std::cout << "Node\tFirst\tLast\n";
        for (int i = 0; i < N; ++i)
        {
            std::cout << "Node " << i << ": " << n_first[i] << "\t" << n_last[i] << "\n";
        }
    #endif
}
} // namespace mce