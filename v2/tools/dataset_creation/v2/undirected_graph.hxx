#ifndef UNDIRECTED_GRAPH_H
#define UNDIRECTED_GRAPH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdexcept>
#include <cassert>
#include <queue>

class undirected_graph {
public:
    std::vector<long> vertices;
    std::vector<int> edges;
    std::vector<int> src;
    std::vector<int> dest;
    std::vector<int> degrees;

    std::vector<int> parent;
    std::vector<int> roots;
    std::vector<bool> visited;
    
    int numVert = 0;
    long numEdges = 0;

    int max_degree_vert = -1;
    double avg_out_degree = 0.0; 

    undirected_graph(const std::string& filename) {
        std::string extension = getFileExtension(filename);
        if (extension == ".txt") {
            readEdgesgraph(filename);
        } else if (extension == ".egr") {
            readECLgraph(filename);
        } else {
            std::cerr << "Unsupported file extension: " << extension << std::endl;
            return;
        }
    }

    void csrToList() {
        src.clear();
        dest.clear();
        src.reserve(numEdges / 2);
        dest.reserve(numEdges / 2);
        degrees.resize(numVert);

        long ctr = 0;

        for (int i = 0; i < numVert; ++i) {
            for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
                if(i < edges[j]) {
                    src.push_back(i);
                    dest.push_back(edges[j]);
                    ctr++;
                }
            }
        }    

        assert(ctr == numEdges/2);

        long max_degree = 0;
        max_degree_vert = -1;
        avg_out_degree = 0.0;
        for (int i = 0; i < numVert; ++i) {
            long degree = vertices[i+1] - vertices[i];
            degrees[i] = degree;
            avg_out_degree += (double)degree;
            if (degree > max_degree) {
                max_degree = degree;
                max_degree_vert = i;
            }
        }
        avg_out_degree /= (double)numVert;
        
        assert(max_degree_vert >= 0);
        assert(avg_out_degree >= 0.0);
    }

    int start_bfs() {
        parent.resize(numVert);
       int numComp = 0;
       visited.resize(numVert, false);

       // Create a random device and seed a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());  // Mersenne Twister generator

        // Define a uniform integer distribution range from 0 to n-1
        std::uniform_int_distribution<> dist(0, numVert - 1);

        // Generate the random root
        int root = dist(gen);
        std::cout << "Chosen root for BFS: " << root << std::endl;

       BFS_CSR(root);
        roots.push_back(root);
        numComp++;

        for (int i = 0; i < numVert; ++i) {
            if (!visited[i]) {
                BFS_CSR(i);
                roots.push_back(i);
                numComp++;
            }
        }
        return numComp;
    }

private:
    void readECLgraph(const std::string& filepath) {
        std::ifstream inFile(filepath, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Error opening file: " + filepath);
        }

        // Reading sizes
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        vertices.resize(size);
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        edges.resize(size);

        // Reading data
        inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
        inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

        numVert = vertices.size() - 1;
        numEdges = edges.size();

        csrToList();
    }

    void readEdgesgraph(const std::string& filepath) {
        std::ifstream inFile(filepath);
        if (!inFile) {
            throw std::runtime_error("Error opening file: " + filepath);
        }
        inFile >> numVert >> numEdges;

        std::vector<std::vector<int>> adjlist(numVert);
        src.reserve(numEdges / 2);
        dest.reserve(numEdges / 2);
        
        int u, v;
        for(long i = 0; i < numEdges; ++i) {
            inFile >> u >> v;
            adjlist[u].push_back(v);
            if(u < v) {
                src.push_back(u);
                dest.push_back(v);
            }
        }

        createCSR(adjlist);  
    }

    void createCSR(const std::vector<std::vector<int>>& adjlist) {
    
        int numVert = adjlist.size();
        degrees.resize(numVert);
        vertices.push_back(edges.size());
        for (int i = 0; i < numVert; i++) {
            edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
            vertices.push_back(edges.size());
        }

        long max_degree = 0;
        max_degree_vert = -1;
        avg_out_degree = 0.0;
        
        for (int i = 0; i < numVert; ++i) {
            long degree = vertices[i+1] - vertices[i];
            degrees[i] = degree;
            avg_out_degree += (double)degree;
            if (degree > max_degree) {
                max_degree = degree;
                max_degree_vert = i;
            }
        }
        avg_out_degree /= (double)numVert;
        
        assert(max_degree_vert >= 0);
        assert(avg_out_degree >= 0.0);
    }

    std::string getFileExtension(const std::string& filename) {
        auto pos = filename.find_last_of(".");
        if (pos != std::string::npos) {
            return filename.substr(pos);
        }
        return "";
    }

    void BFS_CSR(int start) {
        std::queue<int> q;

        q.push(start);
        visited[start] = true;
        parent[start] = start;

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            // Using CSR to iterate over neighbors
            int start_edge = vertices[current];
            int end_edge = (current + 1 < vertices.size()) ? vertices[current + 1] : edges.size();

            for (int edge_idx = start_edge; edge_idx < end_edge; ++edge_idx) {
                int neighbor = edges[edge_idx];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                    parent[neighbor] = current;
                }
            }
        }
    }
};

#endif // UNDIRECTED_GRAPH_H