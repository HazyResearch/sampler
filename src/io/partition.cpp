#include "io/partition.h"
#include <fstream>

Partition::Partition(int _num_partitions, int _num_weights, std::string variable_file, std::string factor_file) 
	: num_partitions(_num_partitions) {
	std::stringstream ss;
	for (int i = 0; i < num_partitions; i++) {
		ss.str("");
		ss << i;
		numbers.push_back(ss.str());
		Meta meta;
		meta.num_weights = _num_weights;
		metas.push_back(meta);

		std::unordered_map<long, long> *map = new std::unordered_map<long, long>;
		vid_maps.push_back(map);
		map = new std::unordered_map<long, long>;
		fid_maps.push_back(map);
	}
	load_mapping(variable_file, variable_pid_map);
	load_mapping(factor_file, factor_pid_map);
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
	std::vector<long> counts = partition(filename, 35, variable_pid_map);
	for (int i = 0; i < num_partitions; i++) {
		metas[i].num_variables = counts[i];
	}
}

void Partition::partition_factors(std::string filename) {
	std::vector<long> counts = partition(filename, 26, factor_pid_map);
	for (int i = 0; i < num_partitions; i++) {
		metas[i].num_factors = counts[i];
	}
}

void Partition::partition_edges(std::string filename) {
	std::vector<long> counts = partition(filename, 33, variable_pid_map);
	for (int i = 0; i < num_partitions; i++) {
		metas[i].num_edges = counts[i];
	}
}

std::vector<long> Partition::partition(std::string filename, int size, std::unordered_map<long, int> map) {
	std::ifstream file;
	std::vector<std::ofstream *> outstreams;
	std::vector<long> counts;
	// write variables to each partition file
	for (int i = 0; i < num_partitions; i++) {
		std::ofstream *of = new std::ofstream();
		of->open((filename + ".part" + numbers[i]).c_str(), std::ios::out | std::ios::binary);
		outstreams.push_back(of);
		counts.push_back(0);
	}
	file.open(filename.c_str(), std::ios::binary);
	long idn, idh; // id in network order and host order
	int pid;
	size = size - 8;
	char *buf = new char[size];
	while (true) {
		if (!file.read((char *)&idn, 8)) break;
		file.read(buf, size);
		idh = bswap_64(idn);
		pid = variable_pid_map[idh];
		outstreams[pid]->write((char *)&idn, 8);
		outstreams[pid]->write(buf, size);
		counts[pid] += 1;
	}
	file.close();
	for (int i = 0; i < num_partitions; i++) {
		outstreams[i]->close();
		delete outstreams[i];
	}
	return counts;
}

Partition::~Partition() {
	for (int i = 0; i < num_partitions; i++) {
		delete vid_maps[i];
		delete fid_maps[i];
	}
}
