#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "serial_rst/bfs.hpp"

#include "common/graph.hpp"
#include "common/cuda_utility.cuh"

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"

#include "cuda_bfs/cuda_bfs.cuh"

// #define DEBUG

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph> <delete_batch>" << std::endl;
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(2), "Failed to set device");
    
    std::string filename = argv[1];
    std::string delete_filename = argv[2];
    
    graph G(filename);

    std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;

    std::vector<int> parent(G.numVert);
    std::vector<int> roots;
    
    std::cout << "\t\tBFS Started...\n";
    int numComp = bfs(G.vertices, G.edges, parent, roots);

    std::cout <<"Number of components in the graph : " << numComp << std::endl;

    #ifdef DEBUG
        // G.print_CSR();
        G.print_list();
        std::cout << "Parent array:\n";
        host_print(parent);
    #endif
        
    // calculate the eulerian tour
    EulerianTour euler_tour(G.numVert);

    dynamic_tree_manager tree_ds;
    tree_ds.read_delete_batch(delete_filename);
    tree_ds.mem_alloc(parent, G.edge_list);
    tree_ds.update_existing_ds();

    std::cout << "The edge list has been updated.\n";

    // baseline_1: bfs
    cuda_BFS(tree_ds.d_updated_edge_list, tree_ds.num_vert, tree_ds.num_edges);

    repair_spanning_tree(roots, tree_ds, euler_tour);

    return EXIT_SUCCESS; 
}
