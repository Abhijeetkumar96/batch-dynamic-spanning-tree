#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "serial_rst/spanning_tree.hpp"

#include "common/graph.hpp"
#include "common/cuda_utility.cuh"
#include "common/Timer.hpp"

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"
#include "dynamic_spanning_tree/euler_tour.cuh"

#include "cuda_bfs/cuda_bfs.cuh"

// #define DEBUG

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
long    maxThreadsPerBlock  = 0;

int validate(int*, int);

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph> <delete_batch>" << std::endl;
        return EXIT_FAILURE;
    }

    cuda_init(2);
    
    std::string filename = argv[1];
    std::string delete_filename;
    bool testgen = false;
    
    if(argc < 3) {
        delete_filename = "delete_edges.txt";
        testgen = true;
    }
    else {
        delete_filename = argv[2];
    }
    graph G(filename, testgen);

    std::cout << "\nfilename: " << extract_filename(filename) << std::endl;
    
    // std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;

    // all baselines: bfs (simple + Adam_Polak + PR_RST)
    cuda_BFS(G, delete_filename);

    print_total_function_time();

    std::cout << "----------------------------------------------------------------" << std::endl;

    return EXIT_SUCCESS; 
}
