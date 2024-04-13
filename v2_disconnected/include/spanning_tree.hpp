#ifndef BFS_H
#define BFS_H

int dfs(const std::vector<std::vector<int>>& adj, std::vector<int>& parent, std::vector<int>& roots);
int bfs(const std::vector<std::vector<int>>& adjlist, std::vector<int>& parent, std::vector<int>& roots);

#endif //bfs.h