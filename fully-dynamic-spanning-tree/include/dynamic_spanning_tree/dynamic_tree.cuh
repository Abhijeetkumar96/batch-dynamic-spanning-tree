#ifndef DYNAMIC_TREE_H
#define DYNAMIC_TREE_H

#include <string>
#include "dynamic_tree_util.cuh"

extern std::string rep_edge_algo;
extern std::string path_rev_algo;

void repair_spanning_tree(dynamic_tree_manager& tree_ds, bool is_deletion = true);

#endif // DYNAMIC_TREE_H