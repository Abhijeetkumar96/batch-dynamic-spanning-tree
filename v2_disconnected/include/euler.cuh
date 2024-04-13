#ifndef EULER_H
#define EULER_H

#include <iostream>


inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

class EulerianTour {
	public:
	    int2 *d_edge_num;
	    int *d_child_count;
	    int *d_child_num;
	    int *starting_index;
	    int *d_successor;
	    int *d_euler_tour_arr;
	    int *d_child_list;
	    int *d_first_edge;
	    int *d_rank;
	    int *d_new_first;
	    int *d_new_last;
		
		int N;
	    int edges;
	    int edge_count;
	    int num_comp;
	    
	    // list_ranking params
	    unsigned long long *devRankNext;
	    int *devNotAllDone;
	    int *notAllDone;

	    // Constructor declaration
	    EulerianTour(int N, int num_comp);
	    // Destructor declaration
	    ~EulerianTour();
	    
	    void mem_alloc();
	    void mem_init();
	};

void euler_tour(int* d_parent, int* d_roots, int* d_rep_arr, int* d_rep_map, int nodes, int num_comp, EulerianTour& euler_mag);

#endif // EULER_H
