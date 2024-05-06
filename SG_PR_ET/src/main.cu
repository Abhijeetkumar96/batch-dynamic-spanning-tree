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

    std::cout << "\n\n";
    std::cout << "filename: " << extract_filename(filename) << std::endl;
    
    // std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;

    g_verbose = false;
    
    std::vector<int> parent(G.numVert);
    std::vector<int> roots;
    
    int root = 0;
    bool _flag = true;

    int numComp = -1;

    if(_flag) {
        // std::cout << "\t\tBFS Started...\n";
        numComp = bfs(G.vertices, G.edges, root, parent, roots);
    }
    else {
        // std::cout << "\t\tDFS Started...\n";
        numComp = dfs(G.vertices, G.edges, root, parent, roots);
    }

    // std::cout <<"Number of components in the input graph : " << numComp << std::endl;

    if(g_verbose) {
        // G.print_CSR();
        G.print_list();
        std::cout << "\nParent array from main function:\n";
        int j = 0;
        for(auto i : parent) 
            std::cout << "parent[" << j++ << "] = " << i << std::endl;
        std::cout << std::endl;
    }

    // g_verbose = false;
    // calculate the eulerian tour
    EulerianTour euler_tour(G.numVert);

    dynamic_tree_manager tree_ds(parent, delete_filename, G.edge_list, root);

    if(g_verbose) {
        std::cout << "updated edgelist from main:\n";
        print_device_edge_list(tree_ds.d_updated_edge_list, tree_ds.num_edges);

        std::cout << "The edge list has been updated.\n";
    }

    repair_spanning_tree(roots, tree_ds, euler_tour);

    print_total_function_time();

    // validate the output
    int* new_parent = tree_ds.new_parent;

    int temp = validate(new_parent, G.numVert);
    std::cout << "numComp after edge deletion: " << temp << std::endl;

    return EXIT_SUCCESS; 
}

int validate(int* parent, int n) {

    std::cout << "Executing validate part.\n";
    
    std::vector<int> h_parent(n);
    CUDA_CHECK(cudaMemcpy(h_parent.data(), parent, n * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back");

    #ifdef DEBUG
        int j = 0;
        for(auto i : h_parent) {
            std::cout << "parent[" << j++ << "]= " << i << "\n";
        }
        std::cout << std::endl;
    #endif
    
    // if(validateRST(h_parent)) {
    //     std::cout<<"Validation successful using validateRST."<<std::endl;
    //     std::cout << "tree depth = " << treeDepth(h_parent) << std::endl;
    // }
    // else {
    //     std::cerr << "Validation failure" << std::endl;
    //     return -1;
    // }

    // std::cout << "Doing pointer_jumping.\n";
    pointer_jumping(parent, n);

    // find_unique(parent, d_out, n, result);

    std::vector<int> h_rep(n);
    CUDA_CHECK(cudaMemcpy(h_rep.data(), parent, sizeof(int) * n, cudaMemcpyDeviceToHost),
        "Failed to copy d_rep array to host");

    std::set<int> unique_elements(h_rep.begin(), h_rep.end());
    int result = unique_elements.size();
    std::cout << "Unique representatives after deleting edges: " << result << "\n";

    return result;
}