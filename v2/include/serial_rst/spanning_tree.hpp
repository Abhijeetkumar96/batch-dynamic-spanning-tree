#ifndef BFS_H
#define BFS_H

int bfs(
	const std::vector<long>& nodes, 
	const std::vector<int>& edges, 
	std::vector<int>& parent, 
	std::vector<int>& roots);

int dfs(
	const std::vector<std::vector<int>>& adj, 
	std::vector<int>& parent, 
	std::vector<int>& roots);

#endif //bfs.h