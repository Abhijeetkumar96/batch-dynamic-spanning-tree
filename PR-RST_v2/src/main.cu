#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <thrust/device_vector.h>

#include "rootedSpanningTreePR.h"
#include "utility.h"

#define DEBUG

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];
	std::ifstream inputFile(filename);
	if (!inputFile)
	{
		std::cerr << "Unable to open the file for reading.\n";
		return EXIT_FAILURE;
	}

	int n, e;
	inputFile >> n >> e;
	int u, v;

	std::vector<uint64_t> h_edgelist;
	
	for (int i = 0; i < e; ++i)
	{
		inputFile >> u >> v;
		// if(u > v) {
			h_edgelist.push_back((uint64_t)(u) << 32 | (v));
		// }
	}

	int numEdges = h_edgelist.size();
	size_t size = numEdges * sizeof(uint64_t);

	uint64_t* d_edgelist;

	cudaMalloc((void **)&d_edgelist, size);

	cudaMemcpy(d_edgelist, h_edgelist.data(), size, cudaMemcpyHostToDevice);

	std::vector<int> parent = RootedSpanningTree(d_edgelist, n, e);
	
	#ifdef DEBUG
		// printArr(parent, n, 10);
		for(auto i : parent)
			std::cout << i <<" ";
		std::cout << std::endl;

	#endif
	
	if(validateRST(parent))
	{
		std::cout<<"Validation success"<<std::endl;
		std::cout << "tree depth = " << treeDepth(parent) << std::endl;
	}
	else
	{
		std::cout<<"Validation failure"<<std::endl;
		exit(1);
	}
}