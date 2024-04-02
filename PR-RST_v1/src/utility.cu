#include "utility.h"

void print(const std::vector<int> &arr)
{
	for (auto i : arr)
		std::cout << i << " ";
	std::cout << std::endl;
}

void print(const thrust::device_vector<int> &arr)
{
	for (auto i : arr)
		std::cout << i << " ";
	std::cout << std::endl;
}

void printArr(const std::vector<int> &parent,int vertices,int stride)
{
	for(int i=0;i<(vertices/stride);i++)
	{
		std::cout<<(i*stride)<<" : ";
		for(int j=0;j<stride;j++)
		{
			if(i*stride+j < parent.size())
				std::cout<<parent[i*stride+j]<<" ";
		}
		std::cout<<"\n";
	}
}

void printPR(const std::vector<int> &pr_arr,const std::vector<int> &pr_arr_size,int vertices,int log_2_size)
{
	std::cout<<"PR arr : \n";
	for(int i=0;i<vertices;i++)
	{
		std::cout<<i<<" : ";
		for(int j=0;j<log_2_size;j++)
		{
			std::cout<<pr_arr[i*log_2_size+j]<<" ";
		}
		std::cout<< " | " <<pr_arr_size[i]<<"\n";
	}
}
