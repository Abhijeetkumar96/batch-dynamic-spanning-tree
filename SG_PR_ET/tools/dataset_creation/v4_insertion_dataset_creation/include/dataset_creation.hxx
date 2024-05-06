#ifndef DATASET_CREATION_H
#define DATASET_CREATION_H

#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "undirected_graph.hxx"

class dataset_creation {

    undirected_graph& g; // Reference to a graph instance for composition
    std::vector<std::pair<int, int>>& bridges;
    std::vector<std::pair<int, int>> tree_edges;
    std::vector<std::pair<int, int>> nonTreeEdges;

	public:
        void non_tree_edges_identify();
        void write(int, const std::string&);
	    dataset_creation(undirected_graph& graph_instance, std::vector<std::pair<int, int>>& cut_edges): g(graph_instance), bridges(cut_edges) { 
            non_tree_edges_identify(); 
            std::cout << "Bridges count: " << cut_edges.size() << std::endl;
        }
};

void dataset_creation::non_tree_edges_identify() {
   
    for(long i = 0; i < g.numEdges / 2; ++i) {
        int u = g.src[i];
        int v = g.dest[i];
        // std::cout << "u: " << u <<" & v: " << v << std::endl;
        // std::cout <<"parent[u]: " << g.parent[u] << "parent[v]: " << g.parent[v] << std::endl; 
        // Check if it's a non-tree edge
        if(g.parent[u] != v and g.parent[v] != u) {
            nonTreeEdges.push_back(std::make_pair(u, v));
        }
    }

    // std::cout << "nonTreeEdges:\n";
    // for(auto i : nonTreeEdges) {
    //     std::cout << i.first <<" " << i.second <<"\n";
    // }
    // std::cout << std::endl;
}

// generate random indices and select those from nonTreeEdges
void dataset_creation::write(int total_edges, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile) {
        std::cerr << "Unable to open the file for writing." << std::endl;
        return;
    }

    // Setup random number generation for percentage
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(40, 70);  // Range for percentage of bridges

    // Generate random percentage between 40% and 70%
    int random_percentage = dis(gen);
    std::cout << "Percent of Bridges to be chosen: " << random_percentage << std::endl;
    double bridge_percentage = random_percentage / 100.0;
    std::cout << "Bridge percent: " << bridge_percentage << std::endl;
    int min_bridges = static_cast<int>(total_edges * bridge_percentage);
    std::cout << "Minimum number of bridges: " << min_bridges << std::endl;

    // Use the actual number of available bridges if less than requested
    int actual_bridges_used = std::min(min_bridges, static_cast<int>(bridges.size()));
    std::cout << "actual bridges to be used: " << actual_bridges_used << "\n";
    
    // Check if there are enough non-tree edges to fill up the rest
    assert(total_edges - actual_bridges_used <= nonTreeEdges.size());

    outputFile << total_edges << "\n";

    // Shuffle bridges to randomize which ones are chosen
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(bridges.begin(), bridges.end(), std::default_random_engine(seed));

    // First, select the actual number of bridges
    for(int i = 0; i < actual_bridges_used; ++i) {
        outputFile << bridges[i].first << " " << bridges[i].second << "\n";
        // std::cout << bridges[i].first << " " << bridges[i].second << "\n";
    }

    // Shuffle non-tree edges to fill the rest of the batch
    std::shuffle(nonTreeEdges.begin(), nonTreeEdges.end(), std::default_random_engine(seed));

    // Fill the rest of the batch with non-tree edges
    for(int i = 0; i < total_edges - actual_bridges_used; ++i) {
        outputFile << nonTreeEdges[i].first << " " << nonTreeEdges[i].second << "\n";
    }

    outputFile.close();
}

#endif // DATASET_CREATION_H