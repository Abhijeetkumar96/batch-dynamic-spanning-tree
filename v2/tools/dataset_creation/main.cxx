/**
 * To compile this program, use the following command:
 * g++ -std=c++17 -o my_program main.cxx -O3
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "dataset_creation.hxx"
#include "undirected_graph.hxx"

// #define DEBUG

std::string output_path = "/raid/graphwork/spanning_tree_datasets/";

std::string get_filename(const std::string path) {
    std::filesystem::path fsPath(path);
    return fsPath.stem().string();
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string filename = argv[1];
    undirected_graph G(filename);

    int nodes = G.numVert;
    long numEdges = G.numEdges;

    std::cout << "numVertices : " << nodes << ", numEdges : " << numEdges << std::endl;

    std::cout << "\t\tBFS Started...\n";
    int numComp = G.start_bfs();

    std::cout <<"Number of components in the graph : " << numComp << std::endl;
    
    dataset_creation data(G);
    // Construct the output file name
    std::string output_path = "/home/cs22s501/spanning_tree/batch-dynamic-spanning-tree/v2/datasets/delete_batch/";
    std::string output_file = output_path + get_filename(filename) + ".txt";

    // Print the output file path for demonstration
    std::cout << "Output file path: " << output_file << "\n\n";

    data.insert_tree_edges(3);
    data.write(7, 3, output_file);

    return EXIT_SUCCESS; 
}

