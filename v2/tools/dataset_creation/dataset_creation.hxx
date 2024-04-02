#ifndef DATASET_CREATION_H
#define DATASET_CREATION_H

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "undirected_graph.hxx"

class dataset_creation {

    undirected_graph& g; // Reference to a graph instance for composition
    std::vector<std::pair<int, int>> tree_edges;
    std::vector<std::pair<int, int>> nonTreeEdges;

	public:
	    dataset_creation(undirected_graph& graph_instance): g(graph_instance) { 
            non_tree_edges_identify(); 
        }
		
		void insert_tree_edges(int);
    	void non_tree_edges_identify();
    	void write(int, int, const std::string&);
};

void dataset_creation::insert_tree_edges(int num_tree_edges_del) {

    // Extract all the edges excluding the root
    for (int i = 0; i < g.numVert; ++i) {
       if (i != g.parent[i]) {  // and g.degrees[i] > g.avg_out_degree
            tree_edges.push_back({i, g.parent[i]});
        }
    }

    // Randomly shuffle the tree_edges
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(tree_edges.begin(), tree_edges.end(), g);

    // Take the first k tree_edges
    tree_edges.resize(num_tree_edges_del);
}

void dataset_creation::non_tree_edges_identify() {
   
    for(long i = 0; i < g.numEdges / 2; ++i) {
        int u = g.src[i];
        int v = g.dest[i];
        std::cout << "u: " << u <<" & v: " << v << std::endl;
        std::cout <<"parent[u]: " << g.parent[u] << "parent[v]: " << g.parent[v] << std::endl; 
        // Check if it's a non-tree edge
        if(g.parent[u] != v and g.parent[v] != u) {
            nonTreeEdges.push_back(std::make_pair(u, v));
        }
    }

    std::cout << "nonTreeEdges:\n";
    for(auto i : nonTreeEdges) {
        std::cout << i.first <<" " << i.second <<"\n";
    }
    std::cout << std::endl;
}

// generate random indices and select those from nonTreeEdges
void dataset_creation::write(int total_edges, int num_tree_edges_del, const std::string& filename) {
    
    std::ofstream outputFile(filename);

    // Check if the file is open
    if (!outputFile) {
    	std::cerr << "Unable to open the file for writing." << std::endl;
    	return;
    }

    outputFile << total_edges << std::endl;
    // Write the tree edges first, followed by nonTree edges
    for(auto i : tree_edges) {
    	outputFile << i.first <<" " << i.second <<"\n";
    }

    // Shuffle non-tree edges
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(nonTreeEdges.begin(), nonTreeEdges.end(), std::default_random_engine(seed));
    int numSelect = total_edges - num_tree_edges_del;

    for(int i = 0; i < numSelect; ++i) {
        std::cout << "Selected non-tree edge: (" << nonTreeEdges[i].first << ", " << nonTreeEdges[i].second << ")" << std::endl;
        outputFile << nonTreeEdges[i].first << " " << nonTreeEdges[i].second <<"\n";
    }

    outputFile << std::endl;
}



#endif // DATASET_CREATION_H