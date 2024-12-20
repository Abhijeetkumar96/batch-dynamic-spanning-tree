#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <queue>
#include <random>
#include <algorithm>

#include "serial_rst/spanning_tree.hpp"

std::vector<std::pair<int, int>> inline selectRandomEdges(const std::vector<int>& parent, int k, const std::vector<int>& degree, const int avg_degree) 
{
    int n = parent.size();

    // Extract all the edges excluding the root
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < n; ++i) {
       if (i != parent[i] and degree[i] > avg_degree) {
            edges.push_back({i, parent[i]});
        }
    }

    // Randomly shuffle the edges
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(edges.begin(), edges.end(), g);

    // Take the first k edges
    edges.resize(k);

    return edges;
}

void inline testcase_generator(const std::vector<int>& parent, int numSelected, const std::vector<int>& degree, int avg_degree) {
    auto selectedEdges = selectRandomEdges(parent, numSelected, degree, avg_degree);
    std::ofstream outputFile("delete_edges.txt");

    // Check if the file is open
    if (outputFile.is_open()) 
    {
        outputFile << numSelected<<std::endl;
        // Write the selected indices to the file
        std::cout << "\nThese edges will get deleted : \n";
        for (const auto& edge : selectedEdges) {
            std::cout << edge.first << " " << edge.second << std::endl;
            outputFile << edge.first << " " << edge.second << std::endl;
        }
        // Close the file
        outputFile.close();

        std::cout << "\n\t...Edges to delete file updated. Thank You..." << std::endl;
    } 
    else {
        std::cout << "Unable to open the file for writing." << std::endl;
    }
}

class graph {
public:
    // csr ds
    std::vector<long> vertices;
    std::vector<int> edges;
    std::vector<int> degree;
    std::string original_filename;
    
    // edge_list
    std::vector<uint64_t> edge_list;
    
    int numVert = 0;
    long numEdges = 0;
    int max_degree = 0;
    int min_degree = 0;
    int median_degree = 0;
    int avg_degree = 0;
    long double var = 0;
    int depth = 0;

    graph(const std::string& filename, bool testgen = false) {
        original_filename = filename;
        std::string extension = getFileExtension(filename);
        if (extension == ".txt") {
            readEdgesgraph(filename);
        } else if (extension == ".egr") {
            readECLgraph(filename);
        } else {
            std::cerr << "Unsupported file extension: " << extension << std::endl;
            return;
        }
        
        if(testgen) {
            std::random_device rd0;
            std::mt19937 gen0(rd0());
            std::uniform_int_distribution<int> dist0(0, numVert - 1);

            int root = dist0(gen0);

            std::vector<int> parent(numVert);
            std::vector<int> roots;
            int numComp = bfs(vertices, edges, root, parent, roots);
            int avg_degree = 0;
            degree.resize(numVert);

            for(int i = 0; i < numVert; ++i) {
                degree[i] = vertices[i+1] - vertices[i];
                avg_degree+= degree[i];
            }

            avg_degree /= numVert;

            int numSelected = static_cast<int>(numVert * 0.20);
            testcase_generator(parent, numSelected, degree, avg_degree);
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
        // std::cout <<"Edge list:\n";
        // for(auto &i : edge_list)
        //     std::cout << i <<" ";
        // std::cout << std::endl;
        std::cout <<"Actual edges:" << std::endl;
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

    void print_stat() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Printing stats.\n";
        std::cout << "----------------------------------------" << std::endl;

        vertex_degrees_var();
        std::cout << "|V| = " << numVert;
        std::cout << "\n|E| = " << numEdges;
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "\ndensity (p): " << density();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nvariance (v) = " << var;
        std::cout << "\nMin degree = " << min_degree;
        std::cout << "\nAvg degree = " << avg_degree;
        std::cout << "\nMedian degree = " << median_degree;
        std::cout << "\nMax degree = " << max_degree;

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

    double density() { 
        return (double)numEdges / (numVert * (numVert - 1.0) / 2.0); 
    }

    void vertex_degrees_var() {
        int n = numVert;
        degree.resize(n);

        // initialize min and max to degree of first vertex
        max_degree = min_degree = vertices[1] - vertices[0];
        for (int v = 0; v < n; ++v) {
            degree[v] = vertices[v+1] - vertices[v];
            avg_degree+= degree[v];
            if (max_degree < degree[v])  
                max_degree = degree[v];
            if (degree[v] < min_degree)  
                min_degree = degree[v];
        }
        
        avg_degree /= numVert;

        var = 0;
        
        for(int i = 0; i < numVert; ++i){
            var += (degree[i] - avg_degree) * 1.0 * (degree[i] - avg_degree);
        }
        var /= (n-1);

        // calculate the median degree
        std::sort(degree.begin(), degree.end());
        if (numVert % 2 == 0) {
            median_degree = (degree[numVert / 2 - 1] + degree[numVert / 2]) / 2.0;
        } else {
            median_degree = degree[numVert / 2];
        }

        return;
    }
};

#endif // GRAPH_H
