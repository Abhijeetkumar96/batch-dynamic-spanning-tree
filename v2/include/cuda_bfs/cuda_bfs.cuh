#ifndef CUDA_BFS
#define CUDA_BFS

#include "common/graph.hpp"

void cuda_BFS(graph& G, const std::string& delete_filename);
void adam_polak_bfs(int numVert, long numEdges, long* d_nodes, int* edges);

#endif // CUDA_BFS