#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <map>

int numberOfComponents(std::vector<int> &par);
int markedComponents(std::vector<int> &marked);
int rootedComponents(std::vector<int> &par);
bool validateRST(const std::vector<int> &parent);
int treeDepth(const std::vector<int> &parent);
bool findEdge(const std::vector<std::pair<int, int>> &edge_stream, std::pair<int, int> target);