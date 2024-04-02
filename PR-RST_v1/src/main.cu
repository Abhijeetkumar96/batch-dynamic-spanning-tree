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
	std::vector<int> u_arr;
	std::vector<int> v_arr;
	
	for (int i = 0; i < e; ++i)
	{
		inputFile >> u >> v;
		// if(u > v) {
			u_arr.push_back(u);
			v_arr.push_back(v);
		// }
	}

#ifdef DEBUG
	std::cout << "Printing edge list : \n";

	for (int i = 0; i < u_arr.size(); ++i) {
		std::cout << i << " : (" << u_arr[i] << ", " << v_arr[i] << ")\n";
	}

#endif
	int numEdges = u_arr.size();
	size_t size = numEdges * sizeof(int);

	int *d_u_arr, *d_v_arr;

	cudaMalloc((void **)&d_u_arr, size);
	cudaMalloc((void **)&d_v_arr, size);

	cudaMemcpy(d_u_arr, u_arr.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v_arr, v_arr.data(), size, cudaMemcpyHostToDevice);

	std::vector<int> parent = RootedSpanningTree(d_u_arr, d_v_arr, n, e);
	
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