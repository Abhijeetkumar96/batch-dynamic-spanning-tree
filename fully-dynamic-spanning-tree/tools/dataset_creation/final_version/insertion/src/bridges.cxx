#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stack>
#include <algorithm>
#include <climits>
#include <queue>
#include <string>
#include <filesystem>

//#define debug 1

using namespace std;

std::string get_filename_without_ext(std::string filename) {
    std::filesystem::path file_path(filename);
    // Extracting the filename without extension
    std::string filename_without_extension = file_path.stem().string();

    return filename_without_extension;
}

std::string get_file_ext(std::string filename) {
    std::filesystem::path file_path(filename);
    // Extracting the file extension
    std::string file_extension = file_path.extension().string();

    return file_extension;
}

/* Graph functions */

class unweightedGraph
{
public:
	int totalVertices;
	long totalEdges;
	std::vector<long> offset;
	std::vector<int> neighbour;
	void read_ecl(std::string filename);
	void read_edges(std::string filename);
public:
	// Graph();
	unweightedGraph(const std::string& filename);
	void printCSR();
};

unweightedGraph::unweightedGraph(const std::string& filename) {
	std::string ext = get_file_ext(filename);
	if (ext == ".edges" || ext == ".eg2" || ext == ".txt") {
        read_edges(filename);
    }
    else if (ext == ".egr" || ext == ".bin" || ".csr") {
        read_ecl(filename);
    } else {
        std::cerr << "Unsupported graph format: " + ext;
    }

}

void unweightedGraph::read_ecl(std::string filename) {
	std::cout <<"Reading ecl file.\n";
	std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file: ";
        return;
    }
    // Reading sizes
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    offset.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    neighbour.resize(size);

    // Reading data
    inFile.read(reinterpret_cast<char*>(offset.data()), offset.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(neighbour.data()), neighbour.size() * sizeof(int));

    totalVertices = offset.size() - 1;
    totalEdges = neighbour.size();
}

void unweightedGraph::read_edges(std::string filename) {
	std::cout <<"Reading edges.\n";
	std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file: ";
    }
    inFile >> totalVertices >> totalEdges;

    std::vector<std::vector<int>> adjlist(totalVertices);
    int u, v;
    for(long i = 0; i < totalEdges; ++i) {
        inFile >> u >> v;
        adjlist[u].push_back(v);
	}
    
    offset.push_back(neighbour.size());
    for (int i = 0; i < totalVertices; i++) {
        neighbour.insert(neighbour.end(), adjlist[i].begin(), adjlist[i].end());
        offset.push_back(neighbour.size());
    }
}

void unweightedGraph::printCSR() {
	cout << "Total Edges = " << totalEdges << endl;
	cout << "Total Vertices = " << totalVertices << endl;
	for (int i = 0; i < totalVertices; i++)
	{
		int u = i;
		cout << "Vertex " << u << " -> { ";
		for (long j = offset[i]; j < offset[i + 1]; j++)
		{
			int v = neighbour[j];
			cout << "( " << u << "," << v << ")";
			cout << ", ";
		}
		cout << "}" << endl;
	}
}

/* DS for DFS */

struct vertex_in_stack {
	int u;
	long ind;
	int parent;
};

struct edge {
	int u, v;
};

void find_bridges(std::string filename, vector<pair<int, int>>& cut_edges) {

	unweightedGraph G(filename);

	// tarjan's algo { iterative approach }

	// DS for iterarions and storing results
	stack<vertex_in_stack> vst;
	int prev_vertex = -1;
	int nchild = 0;
	//vector<bool> visited(G.totalVertices,false);
	vector<int> discovery_time(G.totalVertices, -1);
	vector<int> oldest_vertex(G.totalVertices, INT_MAX);
	int time = 0;
	stack<edge> est;
	int nbcc{}, nce{};
	// queue<edge> cut_edges;
	vector<vector<edge>> bcc;

	// DS for cutvertex
	vector<int> cut_vertex(G.totalVertices, false);

	// initialization
	int root = 0;
	vst.push({ 0,0,-1 });

	while (vst.size()) {
		vertex_in_stack& vs = vst.top();

		int u = vs.u;
		long ind = vs.ind;
		int parent = vs.parent;

		// set the dis time
		if (discovery_time[u] == -1) {
			discovery_time[u] = ++time;
			oldest_vertex[u] = time;
		}

		int bccsize{};
		if (prev_vertex != -1) { // we backtracked
			oldest_vertex[u] = min(oldest_vertex[u], oldest_vertex[prev_vertex]);

			// check if prev_vertex makes u a cut vertex
			if (u != root && oldest_vertex[prev_vertex] >= discovery_time[u]) {
				cut_vertex[u] = true;

				// printing the bcc
				//cout << "\nBCC Comp\n";
				edge e;
				vector<edge> bcc_list;
				++nbcc;
				
				if (est.size() == 0) { // error checking
					cout << "est is zero but it shouldn't be\n";
					#ifdef debug
					cout << "u and prev ver is - " << u << "\t" << prev_vertex << "\n";
					cout << "cutvertex status of u and prev_vertex - " << cut_vertex[u] << "\t" << cut_vertex[prev_vertex];
					#endif
					exit(-1);
				}

				do {
					++bccsize;
					e = est.top();
					edge e2 = e;
					if (e2.v < e2.u) {
						int t = e2.u;
						e2.u = e2.v;
						e2.v = t;
					}
					bcc_list.push_back(e2);
					est.pop();
					//cout << e.u << " " << e.v << "\n";
				} while (e.u != u || e.v != prev_vertex);
				
				bcc.push_back(bcc_list);

				if (est.size() == 0) { // error checking
					cout << "est is emptied by " << u << "\n";
					exit(-1);
				}
			}

			// check if the edge u - prev_vertex is cut edge
			if (oldest_vertex[prev_vertex] > discovery_time[u]) { //cout << "\n" << u << " " << prev_vertex << " is a cut edge\n";
				++nce;

				if (u < prev_vertex) {
					cut_edges.push_back({u, prev_vertex});
				}
				else {
					cut_edges.push_back({prev_vertex, u});
				}
				
				if ( (u != root && bccsize != 1) || (u == root && est.size()!=1) ) {
					cout << "we encountered a trivial bcc yet the stack contains more edges\n";
					exit(-1);
				}
			}

			// clear the backtrac info
			prev_vertex = -1;
		}


		/* now we are moving forward */

		long end_ind = G.offset[u + 1];

		// if we exhausted all neighbouring vertices
		if (ind == end_ind) {
			vst.pop();
			prev_vertex = u;
			if (u == root) { // for root special condition
				//cout << "\nprinting last root bcc\n";
				++nbcc;
				vector<edge> bcc_list;

				while (est.size()) {
					edge e = est.top();
					if (e.v < e.u) {
						int t = e.u;
						e.u = e.v;
						e.v = t;
					}
					bcc_list.push_back(e);
					est.pop();
					//cout << e.u << " " << e.v << "\n";
				}

				bcc.push_back(bcc_list);
			}
			continue;
		}

		int v = G.neighbour[ind];

		if (discovery_time[v] == -1) { // moving forward
			if (u == root) ++nchild;
			vertex_in_stack new_vs;
			new_vs.u = v;
			new_vs.ind = G.offset[v];
			new_vs.parent = u;
			if (u == root && nchild > 1) {
				cut_vertex[root] = true;
				//cout << "\nprinting root bcc\n";
				++nbcc;
				vector<edge> bcc_list;
				while(est.size()) {
					edge e = est.top();
					if (e.v < e.u) {
						int t = e.u;
						e.u = e.v;
						e.v = t;
					}
					bcc_list.push_back(e);
					est.pop();
					//cout << e.u << " " << e.v << "\n";
				}
				bcc.push_back(bcc_list);
			}
			vst.push(new_vs);
			est.push({ u,v }); // pushing onto edge stack
		}
		else if (v != parent && discovery_time[v] < discovery_time[u]) { // or else updating oldest_vertex info if it's a back edge ignoring parent
			oldest_vertex[u] = min(oldest_vertex[u], discovery_time[v]); // updating the old value
			est.push({ u,v });
		}
		vs.ind++;
	}
}