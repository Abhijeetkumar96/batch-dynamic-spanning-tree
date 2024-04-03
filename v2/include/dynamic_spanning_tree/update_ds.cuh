#ifndef UPDATE_DS_H
#define UPDATE_DS_H

void update_edgelist(
    int* d_parent, int num_vert, 
    uint64_t* d_edge_list, uint64_t* d_updated_ed_list, long& num_edges, 
    uint64_t* d_edges_to_delete, int delete_size);

#endif // UPDATE_DS_H