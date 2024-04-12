/**
 * Compilation Instructions:
 * Use the command below to compile this program:
 * g++ -std=c++17 -O3 -o my_program main.cxx
 *
 * Overview:
 * This program generates a breadth-first search (BFS) spanning tree from a randomly selected vertex
 * in a graph. It then evaluates the non-tree edges and selectively removes some of these edges while
 * ensuring the graph remains connected. The aim is to optimize the graph structure by minimizing
 * redundancy without compromising connectivity.
 */


#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "dataset_creation.hxx"
#include "undirected_graph.hxx"

// #define DEBUG

std::string output_path = "/raid/graphwork/spanning_tree_datasets/maybe_connected/";

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

    std::string baseFilename = output_path + get_filename(filename) + "/";

    int nodes = G.numVert;
    long numEdges = G.numEdges;

    std::cout << "numVertices : " << nodes << ", numEdges : " << numEdges << std::endl;

    std::cout << "\t\tBFS Started...\n";
    int numComp = G.start_bfs();

    std::cout <<"Number of components in the graph : " << numComp << std::endl;
    
    std::string delete_file_name;
    dataset_creation data(G);

    // Define the ranges and iterations for each set
    std::vector<int> totals = {100, 1000, 10000, 100000};

    // Outer loop for 5 iterations
    for(int i = 0; i < 5; ++i) {
        // Iterate over each total
        for (size_t totalIndex = 0; totalIndex < totals.size(); ++totalIndex) {
            
            int total = totals[totalIndex];
            std::cout << "delete edges count: " << total << "\n";

                // Construct the directory path
                std::filesystem::path dirPath = std::filesystem::path(baseFilename) / 
                                                ("batch_" + std::to_string(total));
                
                // Create the directory using filesystem
                std::filesystem::create_directories(dirPath);

                // Construct the output file name
                std::string output_file = (dirPath / ("set_" + std::to_string(i) + ".txt")).string();

                // Print the output file path for demonstration
                std::cout << "Output file path: " << output_file << "\n\n";

                data.write(total, output_file);

            }
            std::cout << "\n";
        }

    return EXIT_SUCCESS; 
}
