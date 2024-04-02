#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "graph.hpp"
#include "bfs.hpp"
#include "cuda_utility.cuh"
#include "euler_tour.cuh"

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph>" << std::endl;
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(2), "Failed to set device");
    
    std::string filename = argv[1];
    graph G(filename);

    std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;

    std::vector<int> parent(G.numVert);
    std::vector<int> roots;
    
    std::cout << "\t\tBFS Started...\n";
    int numComp = bfs(G.vertices, G.edges, parent, roots);

    std::cout <<"Number of components in the graph : " << numComp << std::endl;

    #ifdef DEBUG
        G.print_CSR();
        host_print(parent);
    #endif
        
    // calculate the eulerian tour
    EulerianTour euler_tour(G.numVert);

    // dynamic_tree_manager tree_ds();
    // tree_ds.read_delete_batch(delete_filename);
    // tree_ds.mem_alloc(parent, g.edgelist);
    // tree_ds.update_existing_ds();
    
    // cal_first_last(roots[0], d_parent, euler_tour);
    // repair_spanning_tree(tree_ds, euler_tour);

    return EXIT_SUCCESS; 
}
