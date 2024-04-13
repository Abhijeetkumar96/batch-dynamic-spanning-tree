// we are testing for graph_2 as of now
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#include "spanning_tree.hpp"
#include "euler.cuh"
#include "pointerJumping.cuh"

#define DEBUG

__global__ 
void update_mapping_kernel(const int* roots, int* rep_map, int num_roots) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_roots) {
        rep_map[roots[idx]] = idx;
    }
}

__global__ 
void find_roots(const int* parent, int* roots, int* d_num_comp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (parent[idx] == idx) {
            int pos = atomicAdd(d_num_comp, 1); 
            roots[pos] = idx; 
        }
    }
}

void generate_interval(std::vector<int> vert_u, std::vector<int>& h_rep_arr, std::vector<int>& h_rep_map, std::vector<int>& intervals, std::vector<int>& h_first, std::vector<int>& h_last) {

    for(int u : vert_u) {
        int rep_u = h_rep_arr[u];
        int mapped_rep = h_rep_map[rep_u];
        intervals[mapped_rep] = u;
    }

    std::cout << "interval array:\n";
    for(auto interval : intervals)
        std::cout << interval << " ";
    std::cout << std::endl;

    for(int i = 0; i < h_rep_arr.size(); ++i) {

        int x = h_rep_arr[i];
        int y = intervals[h_rep_map[x]];

        std::cout << "starting_node for vertex[" << i << "] : " << x << " and ending_node: " << y << "\n";

        // Print values at these indices from h_first
        std::cout << "h_first[" << x << "]: " << h_first[x] << std::endl;
        std::cout << "h_first[" << i << "]: " << h_first[i] << std::endl;
        std::cout << "h_first[" << y << "]: " << h_first[y] << std::endl;

        // Print values at these indices from h_last
        std::cout << "h_last[" << x << "]: " << h_last[x] << std::endl;
        std::cout << "h_last[" << i << "]: " << h_last[i] << std::endl;
        std::cout << "h_last[" << y << "]: " << h_last[y] << std::endl;

        if((h_first[x] < h_first[i]) && (h_first[i] <= h_first[y]) && (h_last[x] >= h_last[i]) && h_last[y] <= h_last[i]) {
            std::cout << "path of " << i << "needs to be reversed.\n";
        }

        std::cout << std::endl;
    }
}

void reverse_path(std::vector<int>& h_rep_arr, std::vector<int>& h_rep_map, std::vector<int>& h_first , std::vector<int>& h_last, int nodes, int num_comp) {

    std::vector<int> vert_u =  {5, 16, 18};
    std::vector<int> parent_u = {5, 4, 16};

    std::vector<int> h_interval(num_comp);
    // generate interval
    generate_interval(vert_u, h_rep_arr, h_rep_map, h_interval, h_first, h_last);

}

int main(int argc, char* argv[]) {
    std::cout << "\nNote: The input graph is expected to be undirected. "
                 "For every edge (u,v), its counterpart (v,u) must also be present." 
                 << std::endl;

    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    std::ifstream inputFile(filename);
    
    if(!inputFile) {
    	std::cerr << "Error: Could not open file " << argv[1] << std::endl;
    	return EXIT_FAILURE;
    }

    int nodes, edges, u, v;
    inputFile >> nodes >> edges;
    std::cout << "numVertices : " << nodes << ", numEdges : " << edges << std::endl;
    
    std::vector<std::vector<int>> adjlist(nodes);

    for(int i = 0; i < edges; ++i) {
    	inputFile >> u >> v;
    	adjlist[u].push_back(v);
    }

    std::cout << "Reading graph " << filename << " completed. \n";
    
    std::vector<int> parent(nodes);
    std::vector<int> roots;
    std::cout << "\tRST Started...\n";
    int numComp = dfs(adjlist, parent, roots);

    std::cout <<"Number of components in the graph : " << numComp << std::endl;

    #ifdef DEBUG
        std::cout << "parent array:\n";
        int j = 0;
        for(auto i : parent)
            std::cout << "parent[" << j++ << "] = " << i << "\n";
        std::cout << std::endl;

        std::cout << "Serial Output:\n";
        for(auto root : roots)
            std::cout << root << " ";
        std::cout << std::endl;
    #endif
    
    // ************************ memory allocations start ************************

    int size = nodes * sizeof(int);
    int *d_rep_arr, *d_parent, *d_roots, *d_num_comp, *d_rep_map;

    //Allocate GPU memory
    cudaMalloc((void**)&d_rep_map, size);

    cudaMalloc((void**)&d_rep_arr, size);
    cudaMemcpy(d_rep_arr, parent.data(), size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_parent, size);
    cudaMemcpy(d_parent, parent.data(), size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_roots, size);

    cudaMallocManaged((void**)&d_num_comp, sizeof(int));
    cudaMemset(d_num_comp, 0, sizeof(int));

    // ************************ memory allocations over ************************

    int block_size = 1024;
    int num_blocks = (nodes + block_size - 1) / block_size;

    // calculate num_components and identify the roots
    find_roots<<<num_blocks, block_size>>>(d_parent, d_roots, d_num_comp, nodes);
    cudaDeviceSynchronize();

    int num_comp = *d_num_comp;
    EulerianTour euler_mag(nodes, num_comp);
    
    #ifdef DEBUG
        std::vector<int> h_roots(num_comp);
        cudaMemcpy(h_roots.data(), d_roots, num_comp * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Roots found: ";
        for (int i = 0; i < num_comp; i++) {
            std::cout << h_roots[i] << " ";
        }
    #endif

    std::cout << "\nTotal components: " << num_comp << std::endl;

    num_blocks = (num_comp + block_size - 1) / block_size;

    update_mapping_kernel<<<num_blocks, block_size>>>(d_roots, d_rep_map, num_comp);
    cudaDeviceSynchronize();

    // pointer_jumping on parent array gives d_rep_arr.
    pointer_jumping(d_rep_arr, nodes);

    // Find the Euler tour for the graph components.
    euler_tour(d_parent, d_roots, d_rep_arr, d_rep_map, nodes, num_comp, euler_mag);

    #ifdef DEBUG
        // copy back d_rep_arr to host
        std::vector <int> h_rep_arr(nodes);
        cudaMemcpy(h_rep_arr.data(), d_rep_arr, sizeof(int) * nodes, cudaMemcpyDeviceToHost);

        // copy back d_rep_map to host
        std::vector <int> h_rep_map(nodes);
        cudaMemcpy(h_rep_map.data(), d_rep_map, sizeof(int) * nodes, cudaMemcpyDeviceToHost);

        int *d_first = euler_mag.d_new_first;
        int *d_last = euler_mag.d_new_last;

        std::vector<int> h_first(nodes), h_last(nodes);
        cudaMemcpy(h_first.data(), d_first, sizeof(int) * nodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_last.data(), d_last, sizeof(int) * nodes, cudaMemcpyDeviceToHost);

        // print values
        std::cout << "rep array:\n";
        for(auto rep : h_rep_arr)
            std::cout << rep << " ";
        std::cout << std::endl;

        std::cout << "rep map array:\n";
        for(auto rep : h_rep_map)
            std::cout << rep << " ";
        std::cout << std::endl;

        std::cout << "Node\tFirst\tLast\n";
        for (int i = 0; i < nodes; ++i) {
            std::cout << "Node " << i << ": " << h_first[i] << "\t" << h_last[i] << "\n";
        }
    #endif

    reverse_path(h_rep_arr, h_rep_map, h_first, h_last, nodes, num_comp);

    return EXIT_SUCCESS; 
}
