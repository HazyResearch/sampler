#ifndef BINARY_PARSER_H
#define BINARY_PARSER_H

#include <stdlib.h>
#include <unordered_map>
#include "dstruct/factor_graph/factor_graph.h"
using namespace std;

// meta data
typedef struct {
	long long num_weights;
	long long num_variables;
	long long num_factors;
	long long num_edges;
	string weights_file; 
	string variables_file;
	string factors_file;
	string edges_file;
} Meta;

/**
 * Reads meta data from the given file.
 * For reference of factor graph file formats, refer to 
 * deepdive.stanford.edu
 */
Meta read_meta(string meta_file);

/**
 * Loads weights from the given file into the given factor graph
 */
long long read_weights(std::string filename, dd::FactorGraph &fg,
  std::unordered_map<long, long> *wid_map = NULL,
  std::unordered_map<long, long> *wid_reverse_map = NULL);

/**
 * Loads variables from the given file into the given factor graph
 */
long long read_variables(std::string filename, dd::FactorGraph &fg,
  std::unordered_map<long, long> *vid_map = NULL,
  std::unordered_map<long, long> *vid_reverse_map = NULL);

/**
 * Loads factors from the given file into the given factor graph
 */
long long read_factors(std::string filename, dd::FactorGraph &fg,
  std::unordered_map<long, long> *fid_map = NULL);

/**
 * Loads edges from the given file into the given factor graph
 */
long long read_edges(std::string filename, dd::FactorGraph &fg,
  std::unordered_map<long, long> *vid_map = NULL, 
  std::unordered_map<long, long> *fid_map = NULL);

// keep track of mapped id
inline long get_or_insert(std::unordered_map<long, long> *map, long id, long count) {
  std::unordered_map<long, long>::const_iterator got = map->find(id);
  if (got == map->end()) {
    (*map)[id] = count;
    return count;
  } else {
    return got->second;
  }
}

#endif