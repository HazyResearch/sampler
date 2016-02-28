#ifndef BINARY_PARSER_H
#define BINARY_PARSER_H

#include <stdlib.h>
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
long long read_weights(
    string filename, dd::FactorGraph &,
    std::unordered_map<long long, long long> *wid_map = NULL,
    std::unordered_map<long long, long long> *wid_reverse_map = NULL);

/**
 * Loads variables from the given file into the given factor graph
 */
long long read_variables(
    string filename, dd::FactorGraph &,
    std::unordered_map<long long, long long> *vid_map = NULL,
    std::unordered_map<long long, long long> *vid_reverse_map = NULL);

/**
 * Loads factors from the given file into the given factor graph (original mode)
 */
long long read_factors(
    string filename, dd::FactorGraph &,
    std::unordered_map<long long, long long> *vid_map = NULL,
    std::unordered_map<long long, long long> *wid_map = NULL);

/**
 * Loads factors from the given file into the given factor graph (incremental
 * mode)
 */
long long read_factors_inc(string filename, dd::FactorGraph &);

/**
 * Loads edges from the given file into the given factor graph (incremental
 * mode)
 */
long long read_edges_inc(string filename, dd::FactorGraph &);

// keep track of mapped id
inline long long get_or_insert(std::unordered_map<long long, long long> *map,
                               long long id, long long count) {
  std::unordered_map<long long, long long>::const_iterator got = map->find(id);
  if (got == map->end()) {
    (*map)[id] = count;
    return count;
  } else {
    return got->second;
  }
}

#endif