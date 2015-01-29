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

// 64-bit big endian to little endian
# define bswap_64(x) \
     ((((x) & 0xff00000000000000ull) >> 56)                                   \
      | (((x) & 0x00ff000000000000ull) >> 40)                                 \
      | (((x) & 0x0000ff0000000000ull) >> 24)                                 \
      | (((x) & 0x000000ff00000000ull) >> 8)                                  \
      | (((x) & 0x00000000ff000000ull) << 8)                                  \
      | (((x) & 0x0000000000ff0000ull) << 24)                                 \
      | (((x) & 0x000000000000ff00ull) << 40)                                 \
      | (((x) & 0x00000000000000ffull) << 56))

// 16-bit big endian to little endian
#define bswap_16(x) \
     ((unsigned short int) ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8)))

/**
 * Reads meta data from the given file.
 * For reference of factor graph file formats, refer to 
 * deepdive.stanford.edu
 */
Meta read_meta(string meta_file);

/**
 * Loads weights from the given file into the given factor graph
 */
long long read_weights(string filename, dd::FactorGraph &);

/**
 * Loads variables from the given file into the given factor graph
 */
long long read_variables(string filename, dd::FactorGraph &);

/**
 * Loads factors from the given file into the given factor graph
 */
long long read_factors(string filename, dd::FactorGraph &);

/**
 * Loads edges from the given file into the given factor graph
 */
long long read_edges(string filename, dd::FactorGraph &);

#endif