
#include "verifySol.h"

int numberOfComponents(std::vector<int> &par)
{
	std::map<int,int> components;
	for(auto i : par)
	{
		components[i]++;
	}
	return components.size();
}

int markedComponents(std::vector<int> &marked)
{
	int count = 0;
	for(auto i : marked)
	{
		count += (i != -1);
	}
	return count;
}

int rootedComponents(std::vector<int> &par)
{
	int count = 0;
	for (int i = 0; i < (int)par.size(); i++)
	{
		count += (i == par[i]);
	}
	return count;
	
}

bool findEdge(const std::vector<std::pair<int, int>> &edge_stream, std::pair<int, int> target)
{
	return std::find(edge_stream.begin(), edge_stream.end(), target) != edge_stream.end();
}

std::vector<int> computeDepths(const std::vector<int> &parent)
{
	int n = parent.size();
	std::vector<int> depth(n, -1);

	for (int i = 0; i < n; ++i)
	{
		int current_depth = 0;
		int j = i;

		// Follow parent pointers to compute depth
		while (depth[j] == -1 && parent[j] != j)
		{
			j = parent[j];
			current_depth++;
		}

		// Set the depth
		depth[i] = current_depth + depth[j];
	}

	return depth;
}

bool validateRST(const std::vector<int> &parent)
{
	int n = parent.size();
	int cnt_root = 0,root = -1;
	std::vector<int> root_nodes;
	for (int i = 0; i < n; i++)
	{
		if(parent[i] == i)
		{
			cnt_root++;
			root = i;
			root_nodes.push_back(i);
		}
	}

	if(cnt_root == 0)
	{
		std::cout<<"Wrong : No root exist"<<std::endl;
		return false;		
	}

	if(cnt_root != 1)
	{
		std::cout<<"Wrong : Multiple root nodes exist"<<std::endl;
		for(auto i : root_nodes)
		{
			std::cout<<i<<' ';
		}
		std::cout<<std::endl;
		return false;
	}

	std::vector<std::vector<int>> adj(n);
	for (int i = 0; i < n; i++)
	{
		if(i!=root)
			adj[parent[i]].push_back(i);
		// adj[i].push_back(parent[i]);
	}

	std::vector<int> vis(n,0);
	
	std::function<bool(int)> dfs = [&](int src)
	{
		vis[src] = 1;
		for(auto child : adj[src])
		{
			if(vis[child] == 0)
			{
				if(dfs(child))
				{
					return true;
				}
			}
			else if(vis[child] == 1)
			{
				return true;
			}
		}
		vis[src] = 2;
		return false;
	};
	
	if(dfs(root))
	{
		std::cout<<"Wrong : Tree has cycles - node["<<root<<"]"<<std::endl;
		return false;
	}

	// for(auto i : adj)
	// {
	// 	for(auto j : i)
	// 		std::cout<<j<<' ';
	// 	std::cout<<std::endl;
	// }


	for(int i=0;i<n;i++)
	{
		// for(auto i : vis) std::cout<<i<<' ';
		// std::cout<<std::endl;
		if(vis[i] != 2)
		{	
			std::cout<<"Wrong : Tree has cycles - node["<<i<<"]"<<std::endl;
			return false;
		}
	}

	return true;
}

int treeDepth(const std::vector<int> &parent)
{
	std::vector<int> depths = computeDepths(parent);
	return *std::max_element(depths.begin(), depths.end());
}