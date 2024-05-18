#ifndef SUPER_GRAPH_H
#define SUPER_GRAPH_H

#include "dynamic_spanning_tree/dynamic_tree_util.cuh"
#include "PR-RST/pr_rst_util.cuh"

/**
 * Computes replacement edges for the dynamic spanning tree.
 * This function creates a superGraph and derives a rooted tree based on the unique representatives
 * of the graph's vertices. It then updates the dynamic tree manager and resource manager with
 * the new edges that replace those removed during updates to the spanning tree.
 *
 * @param tree_ds Reference to the dynamic tree manager, holding the current state of the dynamic spanning tree.
 * @param resource_mag Reference to the resource manager, responsible for managing graph resources and properties.
 * @param unique_rep_count The count of unique representatives, used in calculating the replacement edges.
 * @param is_deletion set to true for a deletion operation, false for an insertion operation.
 */
void get_replacement_edges(
	dynamic_tree_manager& tree_ds, 
	PR_RST& resource_mag, 
	const int& unique_rep_count,
	bool is_deletion);

#endif // SUPER_GRAPH_H