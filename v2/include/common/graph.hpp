#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <queue>

class graph {
public:
    // csr ds
    std::vector<long> vertices;
    std::vector<int> edges;
    
    // edge_list
    std::vector<uint64_t> edge_list;
    
    int numVert = 0;
    long numEdges = 0;

    graph(const std::string& filename) {
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

    void print_CSR() {
        std::cout << "CSR for graph G:\n";
        for (int i = 0; i < numVert; ++i) {
            std::cout << "Vertex " << i << " is connected to: ";
            for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
                std::cout << edges[j] << " ";
            }
            std::cout << "\n";
        }
    }

    void print_list() {
        std::cout <<"Edge list:\n";
        for(auto &i : edge_list)
            std::cout << i <<" ";
        std::cout << std::endl;
        std::cout <<"Actual edges:\n";
        print_actual_edges();
    }

    void print_actual_edges() {
        int j = 0;
        for(auto i : edge_list) {
            int u = i >> 32;  // Extract higher 32 bits
            int v = i & 0xFFFFFFFF; // Extract lower 32 bits
            printf("Edge[%d] = (%d, %d)\n", j, u, v);
            j++;
        }
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
        edge_list.reserve(numEdges / 2);
        
        uint32_t u, v;
        for(long i = 0; i < numEdges; ++i) {
            inFile >> u >> v;
            adjlist[u].push_back(v);
            if(u < v) {
                edge_list.push_back(((uint64_t)(u) << 32 | (v)));
            }
        }

        createCSR(adjlist);  
    }

    void createCSR(const std::vector<std::vector<int>>& adjlist) {
    
        int numVert = adjlist.size();

        vertices.push_back(edges.size());
        for (int i = 0; i < numVert; i++) {
            edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
            vertices.push_back(edges.size());
        }
    }

    void csrToList() {

        edge_list.reserve(numEdges / 2);
        long ctr = 0;

        for (int i = 0; i < numVert; ++i) {
            for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
                if(i < edges[j]) {
                    uint32_t x = i;
                    uint32_t y = edges[j];
                    edge_list.push_back(((uint64_t)(x) << 32 | (y)));
                    ctr++;
                }
            }
        }    

        assert(ctr == numEdges/2);
    }

    std::string getFileExtension(const std::string& filename) {
        auto pos = filename.find_last_of(".");
        if (pos != std::string::npos) {
            return filename.substr(pos);
        }
        return "";
    }
};

#endif // GRAPH_H