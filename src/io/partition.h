/**
 * Partition the variables and factors
 */

#ifndef PARTITION_H
#define PARTITION_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "io/binary_parser.h"

class Partition {
public:
	std::vector<Meta> metas;
	// numbers (0,1,2,3,...) in string
	std::vector<std::string> numbers;
	// old id -> new id (new id must be continous and 0-based)
	std::vector<std::unordered_map<long, long> *> vid_maps;
	std::vector<std::unordered_map<long, long> *> fid_maps;
	// new id -> old id
	std::vector<std::unordered_map<long, long> *> vid_reverse_map;

	Partition(int _num_partitions, int _num_weights, std::string variable_file, std::string factor_file);
	~Partition();

	// load mapping from the given file
	void load_mapping(std::string filename, std::unordered_map<long, int>& map);

	// partition variables according to partition id
	void partition_variables(std::string filename);

	// partition factors and assign factor ids
	void partition_factors(std::string filename);

	// partition edges
	void partition_edges(std::string filename);

	// partition factor graph
	void partition_factor_graph(std::string variable_file, std::string factor_file, std::string edge_file);

private:
	// underlying the above functions, split file according to partition id
	// returns number of records for each partition
	// size is the size of a record in bytes, use map as id -> mapping 
	std::vector<long> partition(std::string filename, int size, std::unordered_map<long, int> map);

	// id -> partition id map
	std::unordered_map<long, int> variable_pid_map;
	std::unordered_map<long, int> factor_pid_map;

	int num_partitions;
};

#endif