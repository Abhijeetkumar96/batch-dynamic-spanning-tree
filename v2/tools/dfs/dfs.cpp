#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void DFSUtil(int v, const vector<vector<int>>& adjList, vector<bool> &visited, vector<int> &parent) {
    // Mark the current node as visited and print it
    visited[v] = true;
    cout << v << " ";

    // Recur for all the vertices adjacent to this vertex
    for (int adj : adjList[v]) {
        if (!visited[adj]) {
            parent[adj] = v; // Set the parent of the adjacent vertex
            DFSUtil(adj, adjList, visited, parent);
        }
    }
}

void DFS(const int v, vector<int>& parent, const vector<vector<int>>& adjList) {
    
    int numNodes = adjList.size();
    // Initially mark all vertices as not visited
    vector<bool> visited(numNodes, false);
    // Initialize all parents as -1
    parent.resize(numNodes, -1);

    // Call the recursive helper function to print DFS traversal
    DFSUtil(v, adjList, visited, parent);

    // Optionally, print the parent array to see the tree structure
    cout << "\nParent array: \n";
    for(int i = 0; i < numNodes; ++i) {
        cout << "Parent of " << i << " is " << parent[i] << "\n";
    }
}


// ====[ Main Code ]====
int main(int argc, char *argv[]) { 
    
    if (argc < 2) {
        std::cerr << "Error: Missing filename argument." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream inputFile(argv[1]);
    
    if (!inputFile) {
        std::cerr << "Unable to open file " << argv[1];
        return 0;
    }

    std::cout << "Reading the number of nodes and edges." << std::endl;
    int nodes, edges;
    inputFile >> nodes >> edges;
    std::cout << "The number of nodes: " << nodes << " and the number of edges: " << edges << "." << std::endl;
    
    int u, v;
    std::vector<std::vector<int>> adjlist(nodes);

    for(int i = 0; i < edges; ++i){
        inputFile >> u >> v;
        adjlist[u].push_back(v);
    }
        
    std::vector<int> parent;
    int root = 0;
    parent[root] = root;
    DFS(root, parent, adjlist);

    return EXIT_SUCCESS;
}

// ====[ End of Main Code ]====