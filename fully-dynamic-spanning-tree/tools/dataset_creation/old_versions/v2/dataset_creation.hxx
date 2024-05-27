#ifndef DATASET_CREATION_H
#define DATASET_CREATION_H

#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "undirected_graph.hxx"

class dataset_creation {

    undirected_graph& g; // Reference to a graph instance for composition
    std::vector<std::pair<int, int>> tree_edges;
    std::vector<std::pair<int, int>> nonTreeEdges;

	public:
        void non_tree_edges_identify();
        void write(int, const std::string&);
	    dataset_creation(undirected_graph& graph_instance): g(graph_instance) { 
            non_tree_edges_identify(); 
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

    // Check if the file is open
    if (!outputFile) {
    	std::cerr << "Unable to open the file for writing." << std::endl;
    	return;
    }

    // check if enough nonTree edges are there or not
    assert(total_edges < nonTreeEdges.size());

    outputFile << total_edges << std::endl;

    // Shuffle non-tree edges
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(nonTreeEdges.begin(), nonTreeEdges.end(), std::default_random_engine(seed));
    int numSelect = total_edges;

    for(int i = 0; i < numSelect; ++i) {
        // std::cout << "Selected non-tree edge: (" << nonTreeEdges[i].first << ", " << nonTreeEdges[i].second << ")" << std::endl;
        outputFile << nonTreeEdges[i].first << " " << nonTreeEdges[i].second <<"\n";
    }

    outputFile << std::endl;
}



#endif // DATASET_CREATION_H