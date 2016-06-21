#ifndef DIMMWITTED_FACTOR_GRAPH_H_
#define DIMMWITTED_FACTOR_GRAPH_H_

#include "cmd_parser.h"
#include "common.h"

#include "factor.h"
#include "inference_result.h"
#include "variable.h"
#include "weight.h"

#include <xmmintrin.h>

namespace dd {

class FactorGraph;
class CompactFactorGraph;

/** Meta data describing the dimension of a factor graph */
class FactorGraphDescriptor {
 public:
  FactorGraphDescriptor();

  FactorGraphDescriptor(num_variables_t num_variables,
                        num_factors_t num_factors, num_weights_t num_weights,
                        num_edges_t num_edges);

  /** number of all variables */
  num_variables_t num_variables;

  /** number of factors */
  num_factors_t num_factors;
  /** number of edges */
  num_edges_t num_edges;

  /** number of all weights */
  num_weights_t num_weights;

  /** number of evidence variables */
  num_variables_t num_variables_evidence;
  /** number of query variables */
  num_variables_t num_variables_query;
};

std::ostream& operator<<(std::ostream& stream,
                         const FactorGraphDescriptor& size);

/**
 * Class for a (raw) factor graph.
 *
 * This representation of a factor graph is used for saving to and loading from
 * files. Some tools are provided to inspect the binary representation of this
 * factor graph into a more human-friendly TSV format.
 */
class FactorGraph {
 public:
  /** Capacity to allow multi-step loading */
  FactorGraphDescriptor capacity;
  /** Actual count of things */
  FactorGraphDescriptor size;

  // variables, factors, weights
  std::unique_ptr<RawVariable[]> variables;
  std::unique_ptr<RawFactor[]> factors;
  std::unique_ptr<Weight[]> weights;

  /**
   * Constructs a new factor graph with given number number of variables,
   * factors, weights, and edges
   */
  FactorGraph(const FactorGraphDescriptor& capacity);

  /**
   * Loads the corresponding array of objects (weights, variables, factors,
   * domains) into the factor graph.
   *
   * From some old documentation, seems like the proper way to do this is:
   *   1. read variables
   *   2. read weights
   *   3. sort variables and weights by id
   *   4. read factors
   *
   * Where Step 3 seems to be obsolete, since we're reading in variables and
   * weights that are in order.
   */
  void load_weights(const std::string& filename);
  void load_variables(const std::string& filename);
  void load_factors(const std::string& filename);
  void load_domains(const std::string& filename);

  /**
   * Checks whether the edge-based store is correct
   */
  void safety_check();
};

inline std::ostream& operator<<(std::ostream& out, FactorGraph const& fg) {
  out << "FactorGraph(";
  out << fg.size << " / " << fg.capacity;
  out << ")";
  return out;
}

/**
 * A factor graph represented in a way more suitable for Gibbs sampling.
 *
 * This class also provides facilities to persist an instance of a
 * CompactFactorGraph into disk for the purposes of checkpointing and resuming
 * a Gibbs sampling.
 */
class CompactFactorGraph {
 private:
  const std::string snapshot_filename = "graph.checkpoint";

 public:
  FactorGraphDescriptor size;

  std::unique_ptr<Variable[]> variables;
  std::unique_ptr<Factor[]> factors;

  /*
   * For each edge, we store the factor, weight id, factor id, and the
   * variable, in the same index of seperate arrays. The edges are ordered so
   * that the edges for a variable is in a continuous region (sequentially).
   *
   * This allows us to access factors given variables, and access variables
   * given factors faster.
   */
  std::unique_ptr<CompactFactor[]> compact_factors;
  std::unique_ptr<weight_id_t[]> compact_factors_weightids;
  std::unique_ptr<factor_id_t[]> factor_ids;
  std::unique_ptr<VariableInFactor[]> vifs;

  /**
   * Compiles given factor graph into a compact format that's more appropriate
   * for inference and learning.
   *
   * Since the original factor graph initializes the new factor graph,
   * it also has to transfer the variable, factor, and weight counts,
   * and other statistics as well.
   */
  CompactFactorGraph(const FactorGraph& fg);

  /* Constructor to load from binary snapshot files */
  CompactFactorGraph(const FactorGraphDescriptor& size);

  // copy constructor
  CompactFactorGraph(const CompactFactorGraph& other);

  /*
   * Given a factor and variable assignment, returns corresponding categorical
   * factor weight id, using proposal value for the variable with id vid.
   * For categorical weights, for each possible assignment, there is a
   * corresponding
   * indicator function and weight.
   */
  weight_id_t get_categorical_weight_id(const variable_value_t assignments[],
                                        const CompactFactor& fs,
                                        variable_id_t vid,
                                        variable_value_t proposal);

  /**
   * Dumps the factor graph in binary format.
   *
   * The binary snapshot files are stored in the snapshot directory path. A
   * path to a directory, instead of a regular file must be supplied to
   * anticipate needing to use multiple files for the purposes of resuming and
   * checkpointing using mmap().
   *
   * @param[in] snapshot_path
   *   The path to the snapshot directory. If it doesn't exist, it will be
   *   created.
   */
  void dump(const std::string& snapshot_path);

  /**
   * The inverse operation of checkpoint.
   *
   * @param[in] snapshot_path
   *   The path to the snapshot directory. If it doesn't exist, it will be
   *   created.
   *
   * @see CompactFactorGraph::checkpoint.
   */
  void resume(const std::string& snapshot_path);

  /**
   * Given a variable, updates the weights associated with the factors that
   * connect to the variable.
   *
   * Used in learning phase, after sampling one variable,
   * update corresponding weights (stochastic gradient descent).
   */
  void update_weight(const Variable& variable, InferenceResult& infrs,
                     double stepsize);

  /**
   * Returns potential of the given factor
   *
   * does_change_evid = true, use the free assignment. Otherwise, use the
   * evid assignement.
   */
  inline double potential(const CompactFactor& factor,
                          const variable_value_t assignments[]);

  /**
   * Returns log-linear weighted potential of the all factors for the given
   * variable using the propsal value.
   *
   * does_change_evid = true, use the free assignment. Otherwise, use the
   * evid assignement.
   */
  inline double potential(const Variable& variable,
                          const variable_value_t proposal,
                          const variable_value_t assignments[],
                          const weight_value_t weight_values[]);
};

inline double CompactFactorGraph::potential(
    const CompactFactor& factor, const variable_value_t assignments[]) {
  return factor.potential(vifs.get(), assignments, -1, -1);
}

inline double CompactFactorGraph::potential(
    const Variable& variable, variable_value_t proposal,
    const variable_value_t assignments[],
    const weight_value_t weight_values[]) {
  // potential
  double pot = 0.0;
  // pointer to the first factor the given variable connects to
  // the factors that the given variable connects to are stored in a continuous
  // region of the array
  CompactFactor* const fs = &compact_factors[variable.n_start_i_factors];
  switch (variable.domain_type) {
    case DTYPE_BOOLEAN: {
      // the weights, corresponding to the factor with the same index
      const weight_id_t* const ws =
          &compact_factors_weightids[variable.n_start_i_factors];
      // for all factors that the variable connects to, calculate the
      // weighted potential
      for (factor_id_t i = 0; i < variable.n_factors; ++i) {
        weight_id_t wid = ws[i];
        pot += weight_values[wid] *
               fs[i].potential(vifs.get(), assignments, variable.id, proposal);
      }
      break;
    }
    case DTYPE_CATEGORICAL: {
      for (factor_id_t i = 0; i < variable.n_factors; ++i) {
        // get weight id associated with this factor and variable
        // assignment
        weight_id_t wid = get_categorical_weight_id(assignments, fs[i],
                                                    variable.id, proposal);
        if (wid == Weight::INVALID_ID) continue;
        pot += weight_values[wid] *
               fs[i].potential(vifs.get(), assignments, variable.id, proposal);
      }
      break;
    }
    default:
      std::abort();
  }
  return pot;
}

// sort variable in factor by their position
bool compare_position(const VariableInFactor& x, const VariableInFactor& y);

}  // namespace dd

#endif  // DIMMWITTED_FACTOR_GRAPH_H_
