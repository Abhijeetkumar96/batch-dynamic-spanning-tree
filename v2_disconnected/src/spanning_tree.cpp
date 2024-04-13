#include <vector>
#include <stack>
#include <queue>
#include <iostream>

#include "spanning_tree.hpp"

// Perform BFS and update the parent array
void BFS_Kernel(const std::vector<std::vector<int>>& adjlist, int start, std::vector<int>& parent, std::vector<bool>& visited) {
    
    std::queue<int> q;

    q.push(start);
    visited[start] = true;
    parent[start] = start;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int neighbor : adjlist[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
                parent[neighbor] = current;
            }
        }
    }
}

int bfs(const std::vector<std::vector<int>>& adjlist, std::vector<int>& parent, std::vector<int>& roots) {
    int n = adjlist.size(), numComp = 0;
    std::vector<bool> visited(n, false);
   
    // #ifdef TEST
    //     BFS_Kernel(adjlist, 2, parent, visited);
    //     roots.push_back(2);
    // #endif

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) 
        {
            BFS_Kernel(adjlist, i, parent, visited);
            roots.push_back(i);
            numComp++;
        }
    }
    return numComp;
}

void DFS_kernel(int start, const std::vector<std::vector<int>>& adj, std::vector<int>& parent, std::vector<bool>& visited) {
    std::stack<int> stack;

    // Start from the given start vertex
    stack.push(start);
    parent[start] = start;  // Mark the start node's parent as itself (or could be -1 if preferred)

    while (!stack.empty()) {
        int v = stack.top();
        stack.pop();

        if (!visited[v]) {
            visited[v] = true;
            // std::cout << "Vertex: " << v << " - Parent: " << parent[v] << std::endl;

            // Process all adjacent vertices
            for (auto i = adj[v].rbegin(); i != adj[v].rend(); ++i) {
                if (!visited[*i]) {
                    stack.push(*i);
                    parent[*i] = v;  // Set the parent of vertex *i to v
                }
            }
        }
    }
}

int dfs(const std::vector<std::vector<int>>& adj, std::vector<int>& parent, std::vector<int>& roots) {
    int n = adj.size(), numComp = 0;
    std::vector<bool> visited(n, false);

        DFS_kernel(5, adj, parent, visited);
        roots.push_back(5);
        numComp++;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            DFS_kernel(i, adj, parent, visited);
            roots.push_back(i);
            numComp++;
        }
    }
    return numComp;
}
