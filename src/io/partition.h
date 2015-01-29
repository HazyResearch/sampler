/**
 * Partition the variables and factors
 */

#ifndef PARTITION_H
#define PARTITION_H

#include <iostream>
#include <unordered_map>
#include <vector>

class Partition {
public:

	Partition(int _num_partitions, std::string variable_file, std::string factor_file);

	// load mapping from the given file
	void load_mapping(std::string filename, std::unordered_map<long, int>& map);

	// partition variables according to partition id
	void partition_variables(std::string filename);

	// partition factors and assign factor ids
	void partition_factors(std::string filename);

	// partition edges
	void partition_edges(std::string filename);

private:
	// underlying the above functions, split file according to partition id
	// size is the size of a record in bytes, use map as id -> mapping 
	void partition(std::string filename, int size, std::unordered_map<long, int> map);

	// variable id -> partition id map
	std::unordered_map<long, int> variable_map;
	std::unordered_map<long, int> factor_map;
	int num_partitions;
	
	// numbers (0,1,2,3,...) in string
	std::vector<std::string> numbers;
};

#endif