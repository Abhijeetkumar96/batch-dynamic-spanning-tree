#ifndef BFS_H
#define BFS_H

int bfs(
	const std::vector<long>& nodes, 
	const std::vector<int>& edges, 
	int root,
	std::vector<int>& parent, 
	std::vector<int>& roots);

int dfs(const std::vector<long>& nodes,
    const std::vector<int>& edges, 
    int root,
    std::vector<int>& parent, 
    std::vector<int>& roots)

#endif //bfs.h