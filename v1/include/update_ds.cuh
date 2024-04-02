#ifndef UPDATE_DS_H
#define UPDATE_DS_H

void update_existing_ds(int* d_parent, int* d_rep, int num_vert, uint64_t* d_edge_list, long num_edges, uint64_t* d_edges_to_delete, int delete_size);

#endif // UPDATE_DS_H