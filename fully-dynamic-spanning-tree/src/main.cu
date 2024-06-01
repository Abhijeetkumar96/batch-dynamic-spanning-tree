#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

#include "serial_rst/spanning_tree.hpp"

#include "common/graph.hpp"
#include "common/Timer.hpp"
#include "common/cuda_utility.cuh"
#include "common/commandLineParser.cuh"

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "dynamic_spanning_tree/dynamic_tree.cuh"

#define DEBUG

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
long    maxThreadsPerBlock  = 0;

std::string rep_edge_algo;
std::string path_rev_algo;
//---------------------------------------------------------------------

int validate(int*, int);

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // Initialize command line parser and retrieve parsed arguments.
    CommandLineParser cmdParser(argc, argv);
    const auto& args = cmdParser.getArgs();

    if (args.error) {
        std::cerr << CommandLineParser::help_msg << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_init(args.cudaDevice);

    if (args.verbose) {
        std::cout << "Verbose mode enabled." << std::endl;
        g_verbose = true;
    }

    if (args.checkerMode) {
        std::cout << "Checker enabled." << std::endl;
    }

    std::string filename = args.inputFile;
    bool testgen = args.testgen;

    std::string delete_filename = testgen ? "delete_edges.txt" : args.batchInputFile;

    graph G(filename, testgen);

    std::cout << "\n\n";
    std::cout << "filename: " << extract_filename(filename) << std::endl;
    
    if(args.print_stat) {
        G.print_stat();
    }
    
    // Handle replacement edge algorithms
    switch (args.rep_algorithm) {
        case CommandLineParser::SUPER_GRAPH:
            rep_edge_algo = "SG_PR";
            break;
        case CommandLineParser::HOOKING_SHORTCUTTING:
            rep_edge_algo = "HS_ET";
            break;
        default:
            std::cerr << "No valid replacement edge algorithm selected." << std::endl;
            return EXIT_FAILURE;
    }

    // Handle path reversal algorithms
    switch (args.pr_algorithm) {
        case CommandLineParser::EULERIAN_TOUR:
            path_rev_algo = "ET";
            break;
        case CommandLineParser::PATH_REVERSAL:
            path_rev_algo = "PR";
            break;
        default:
            std::cerr << "No valid path reversal algorithm selected." << std::endl;
            return EXIT_FAILURE;
    }

    // std::cout << "numVertices: " << G.numVert << ", numEdges: " << G.numEdges << std::endl;

    g_verbose = true;
    
    std::vector<int> parent(G.numVert);
    std::vector<int> roots;

    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, G.numVert - 1); // Define the range
    int root = distr(gen); // Generate a random number within the range

    root = 2;
    std::cout << "Root: " << root << std::endl;
    bool _flag = true;

    int numComp = -1;

    if(_flag) {
        std::cout << "The input parent array is a BFS tree.\n";
        numComp = bfs(G.vertices, G.edges, root, parent, roots);
    }
    else {
        std::cout << "The input parent array is a DFS tree.\n";
        numComp = dfs(G.vertices, G.edges, root, parent, roots);
    }

    if(g_verbose) {
        // G.print_CSR();
        G.print_list();
        std::cout << "\nParent array from main function:\n";
        int j = 0;
        for(auto i : parent) 
            std::cout << "parent[" << j++ << "] = " << i << std::endl;
        std::cout << std::endl;
    }

    dynamic_tree_manager tree_ds(parent, delete_filename, G.edge_list, root);

    if(g_verbose) {
        std::cout << "updated edgelist from main:\n";
        print_device_edge_list(tree_ds.d_updated_edge_list, tree_ds.num_edges);

        std::cout << "The edge list has been updated.\n";
    }

    repair_spanning_tree(tree_ds);

    print_total_function_time("Deletion");
    reset_function_times();
    int* new_parent = tree_ds.new_parent;
    
    tree_ds.destroy_hashtable_();
    tree_ds.create_hashtable_();
    
    // validate the output
    repair_spanning_tree(tree_ds, false);
    // std::cout << "After repairing:\n";
    print_total_function_time("Insertion");

    int temp = validate(new_parent, G.numVert);
    
    std::cout << "numComp in the graph: " << temp << std::endl;

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

    return result;
}