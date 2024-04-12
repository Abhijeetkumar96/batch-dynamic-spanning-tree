#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "serial_rst/spanning_tree.hpp"

#include "common/graph.hpp"
#include "common/cuda_utility.cuh"

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"

#include "cuda_bfs/cuda_bfs.cuh"

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
long    maxThreadsPerBlock  = 0;

int validate(const int*, int);

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph> <delete_batch>" << std::endl;
        return EXIT_FAILURE;
    }

    cuda_init(2);
    
    std::string filename = argv[1];
    std::string delete_filename = argv[2];
    
    graph G(filename);

    std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;

    // baseline_1: bfs (simple + Adam_Polak)
    cuda_BFS(G, delete_filename);

    g_verbose = false;
    
    std::vector<int> parent(G.numVert);
    std::vector<int> roots;
    
    std::cout << "\t\tBFS Started...\n";
    int numComp = bfs(G.vertices, G.edges, parent, roots);

    std::cout <<"Number of components in the graph : " << numComp << std::endl;

    if(g_verbose) {
        // G.print_CSR();
        G.print_list();
        std::cout << "\nParent array from main function:\n";
        int j = 0;
        for(auto i : parent) 
            std::cout << "parent[" << j++ << "] = " << i << std::endl;
        std::cout << std::endl;
    }
        
    // calculate the eulerian tour
    EulerianTour euler_tour(G.numVert);

    dynamic_tree_manager tree_ds(parent, delete_filename, G.edge_list);

    // if(g_verbose) {
    //     std::cout << "updated edgelist from main:\n";
    //     print_device_edge_list(tree_ds.d_updated_edge_list, tree_ds.num_edges);

    //     std::cout << "The edge list has been updated.\n";
    // }

    repair_spanning_tree(roots, tree_ds, euler_tour);

    // validate the output
    int* new_parent = tree_ds.new_parent;

    int temp = validate(new_parent, G.numVert);
    std::cout << "numComp after edge deletion: " << temp << std::endl;

    return EXIT_SUCCESS; 
}

int validate(const int* parent, int n) {

    std::cout << "Executing validate part.\n";
    int* new_parent;
    CUDA_CHECK(cudaMalloc(&new_parent, n * sizeof(int)), "Failed to allocate memory for new_parent");
    CUDA_CHECK(cudaMemcpy(new_parent, parent, n * sizeof(int), cudaMemcpyDeviceToDevice), "Failed to copy parent array");

    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)), "Failed to allocate memory for d_out");

    std::cout << "Doing pointer_jumping.\n";
    pointer_jumping(new_parent, n);
    int result;
    find_unique(new_parent, d_out, n, result);

    // CUDA_CHECK(cudaFree(new_parent), "Failed to free new_parent");
    // CUDA_CHECK(cudaFree(d_out), "Failed to free d_out");

    return result;
}