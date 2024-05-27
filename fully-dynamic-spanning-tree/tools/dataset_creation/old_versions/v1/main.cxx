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

std::string output_path = "/raid/graphwork/spanning_tree_datasets/normal/";

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
    
    #ifdef DEBUG
        for(int i = 0; i < G.numEdges/2; ++i) {
            std::cout << G.src[i] <<" " << G.dest[i] << "\n";
        }

        std::string output_file = "/home/cse/Desktop/cs22s501/spanning_tree/dataset_creation/ipdps_batch_0.txt";
            std::cout << "Output file path: " << output_file << "\n\n";
            data.insert_tree_edges(5);
            data.write(10, 5, output_file);
    #endif

    // Define the ranges and iterations for each set
    std::vector<int> totals = {100, 1000, 10000, 100000};
    std::vector<std::vector<int>> counts = {
        {5, 10, 50, 100},         // For 100
        {10, 100, 500, 1000},     // For 1000
        {10, 100, 500, 1000, 5000, 10000}, // For 10000
        {10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000} 
    };

    // Outer loop for 5 iterations
    for(int i = 0; i < 5; ++i) {
        // Iterate over each total
        for (size_t totalIndex = 0; totalIndex < totals.size(); ++totalIndex) {
            int total = totals[totalIndex];
            std::cout << total << ":\n";
            
            // Iterate over each count within the current total
            for (size_t countIndex = 0; countIndex < counts[totalIndex].size(); ++countIndex) {
                int delete_tree_edges_count = counts[totalIndex][countIndex];
                int delete_non_tree_edges_count = total - delete_tree_edges_count;

                std::cout << "delete_tree_edges_count: " << delete_tree_edges_count
                          << ", delete_non_tree_edges_count: " 
                          << delete_non_tree_edges_count << "\n";

                // Construct the directory path
                std::filesystem::path dirPath = std::filesystem::path(baseFilename) / 
                                                ("batch_" + std::to_string(total)) / 
                                                (std::to_string(delete_tree_edges_count) + "_tree_edges");
                
                // Create the directory using filesystem
                std::filesystem::create_directories(dirPath);

                // Construct the output file name
                std::string output_file = (dirPath / ("set_" + std::to_string(i) + ".txt")).string();

                // Print the output file path for demonstration
                std::cout << "Output file path: " << output_file << "\n\n";

                data.insert_tree_edges(delete_tree_edges_count);
                data.write(total, delete_tree_edges_count, output_file);

            }
            std::cout << "\n"; 
        }    
    }

    return EXIT_SUCCESS; 
}
