#ifndef PR_RST_UTIL_H
#define PR_RST_UTIL_H

#include <iostream>
#include <cmath>

class PR_RST {

public:
	int *d_winner_ptr;
	int *d_ptr;
	int *d_parent_ptr;
	int *d_new_parent_ptr;
	int *d_pr_arr;
	int *d_OnPath;
	int *d_new_OnPath;
	int *d_marked_parent;
	int *d_next;
	int *d_new_next;
	int *d_index_ptr;
	int *d_pr_size_ptr;
	int *d_flag;

	int num_vert;
	int num_edges;
	int log_2_size;
	long long pr_size;

	PR_RST(int num_vert, int num_edges);
	void mem_alloc();
	void mem_init();
	~PR_RST();
};

#endif // PR_RST_UTIL_H