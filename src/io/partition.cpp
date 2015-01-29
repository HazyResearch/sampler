#include "io/partition.h"
#include "binary_parser.h"
#include <fstream>

Partition::Partition(int _num_partitions, std::string variable_file, std::string factor_file) 
	: num_partitions(_num_partitions) {
	std::stringstream ss;
	for (int i = 0; i < num_partitions; i++) {
		ss.str("");
		ss << i;
		numbers.push_back(ss.str());
	}
	load_mapping(variable_file, variable_map);
	load_mapping(factor_file, factor_map);
}

void Partition::load_meta(std::string filename) {
	std::ifstream file;
	file.open(filename.c_str());
	file >> num_partitions;
	file.close();
}

void Partition::load_mapping(std::string filename, std::unordered_map<long, int>& map) {
	std::ifstream file;
	file.open(filename.c_str());
	long id;
	long pid;
	while (file >> id >> pid) {
		map[id] = pid;					
	}
	file.close();
}

void Partition::partition_variables(std::string filename) {
	std::ifstream file;
	std::vector<std::ofstream *> outstreams;
	// write variables to each partition file
	for (int i = 0; i < num_partitions; i++) {
		std::ofstream *of = new std::ofstream();
		of->open((filename + ".part" + numbers[i]).c_str(), ios::out | ios::binary);
		outstreams.push_back(of);
	}
	file.open(filename.c_str(), ios::binary);
	long idn, idh; // id in network order and host order
	int pid;
	char buf[27];
	while (true) {
		if (!file.read((char *)&idn, 8)) break;
		file.read(buf, 27);
		idh = bswap_64(idn);
		pid = variable_map[idh];
		outstreams[pid]->write((char *)&idn, 8);
		outstreams[pid]->write(buf, 27);
	}
	file.close();
	for (int i = 0; i < num_partitions; i++) {
		outstreams[i]->close();
		delete outstreams[i];
	}
}


