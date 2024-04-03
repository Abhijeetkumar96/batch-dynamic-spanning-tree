#ifndef DYNAMIC_TREE_H
#define DYNAMIC_TREE_H

#include "euler_tour.cuh"
#include "dynamic_tree_util.cuh"

void repair_spanning_tree(const std::vector<int>& roots, dynamic_tree_manager& tree_ds, EulerianTour& euler_tour);

#endif // DYNAMIC_TREE_H