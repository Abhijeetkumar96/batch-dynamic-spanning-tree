#include <vector>
#include <queue>

#include "serial_rst/bfs.hpp"

void BFS_CSR(
    const std::vector<long>& nodes, 
    const std::vector<int>& edges, 
    int start, 
    std::vector<int>& parent, 
    std::vector<bool>& visited) 
{
    std::queue<int> q;

    q.push(start);
    visited[start] = true;
    parent[start] = start;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        // Using CSR to iterate over neighbors
        int start_edge = nodes[current];
        int end_edge = (current + 1 < nodes.size()) ? nodes[current + 1] : edges.size();

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

int bfs(const std::vector<long>& nodes, const std::vector<int>& edges, std::vector<int>& parent, std::vector<int>& roots) {
    int n = nodes.size() - 1, numComp = 0;
    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) 
        {
            BFS_CSR(nodes, edges, i, parent, visited);
            roots.push_back(i);
            numComp++;
        }
    }
    return numComp;
}
