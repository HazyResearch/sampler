#ifndef BINARY_PARSER_H
#define BINARY_PARSER_H

#include <stdlib.h>
#include "dstruct/factor_graph/factor_graph.h"

/**
 * Reads meta data from the given file.
 * For reference of factor graph file formats, refer to 
 * deepdive.stanford.edu
 */
Meta read_meta(string meta_file);

/**
 * Loads weights from the given file into the given factor graph
 */
long long read_weights(string filename, dd::FactorGraph &, long long n_weight);

/**
 * Loads variables from the given file into the given factor graph
 */
long long read_variables(string filename, dd::FactorGraph &, long long n_variable);

/**
 * Loads factors from the given file into the given factor graph (original mode)
 */
long long read_factors(string filename, dd::FactorGraph &, long long n_factor, long long n_edge);

/**
 * Loads factors from the given file into the given factor graph (incremental mode)
 */
long long read_factors_inc(string filename, dd::FactorGraph &, long long n_factor);

/**
 * Loads edges from the given file into the given factor graph (incremental mode)
 */
long long read_edges_inc(string filename, dd::FactorGraph &, long long n_edge);

#endif