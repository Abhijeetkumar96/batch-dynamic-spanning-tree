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
		if(u > v) {
			u_arr.push_back(u);
			v_arr.push_back(v);
		}
	}

#ifdef DEBUG
	std::cout << "Printing edge list : \n";

	for (int i = 0; i < u_arr.size(); ++i) {
		std::cout << i << " : (" << u_arr[i] << ", " << v_arr[i] << ")\n";
	}

#endif

	std::vector<int> parent = RootedSpanningTree(u_arr, v_arr, n);
	
	#ifdef DEBUG
		printArr(parent, n, 10);
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