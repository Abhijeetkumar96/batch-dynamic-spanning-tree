#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "bfs.h"
#include "cuda_utility.cuh"
#include "euler_tour.cuh"

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph>" << std::endl;
        return EXIT_FAILURE;
    }
    cudaSetDevice(2);
    std::string filename = argv[1];

    int nodes;
    long numEdges;

    std::vector<long> vertices;
    std::vector<int> edges;
    
    readGraph(filename, nodes, numEdges, vertices, edges);

    std::cout << "numVertices : " << nodes << ", numEdges : " << numEdges << std::endl;

    std::vector<int> source;
    std::vector<int> destination;

    std::vector<int> parent(nodes);
    std::vector<int> roots;
    std::cout << "\t\tBFS Started...\n";
    int numComp = bfs(vertices, edges, parent, roots);

    std::cout <<"Number of components in the graph : " << numComp << std::endl;
    
    // calculate the eulerian tour
    EulerianTour euler_tour(nodes);

    dynamic_tree_manager tree_ds();
    tree_ds.read_delete_batch(delete_filename);
    tree_ds.mem_alloc(parent, h_edgelist);
    tree_ds.update_existing_ds();
    
    // cal_first_last(roots[0], d_parent, euler_tour);
    repair_spanning_tree(tree_ds, euler_tour);

    return EXIT_SUCCESS; 
}
