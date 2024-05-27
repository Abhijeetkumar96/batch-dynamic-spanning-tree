#ifndef HOOKING_SHORT_CUH
#define HOOKING_SHORT_CUH

#include "repl_edges/repl_edges.cuh"
#include "dynamic_spanning_tree/dynamic_tree_util.cuh"

void hooking_shortcutting(dynamic_tree_manager& tree_ds, REP_EDGES& rep_edge_mag, bool is_deletion);

#endif // HOOKING_SHORT_CUH