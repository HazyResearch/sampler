#ifndef DIMMWITTED_INFERENCE_RESULT_H_
#define DIMMWITTED_INFERENCE_RESULT_H_

#include "cmd_parser.h"
#include "variable.h"
#include "weight.h"
#include <memory>

namespace dd {

class CompactFactorGraph;

/**
 * Encapsulates inference result statistics
 */
class InferenceResult {
 private:
  const CompactFactorGraph &fg;
  const CmdParser &opts;
  size_t weight_values_normalizer;

 public:
  num_variables_t nvars;   // number of variables
  num_weights_t nweights;  // number of weights
  num_tallies_t ntallies;

  std::unique_ptr<num_samples_t[]>
      categorical_tallies;  // this might be slow...

  // array of sum of samples for each variable
  std::unique_ptr<variable_value_t[]> agg_means;
  // array of number of samples for each variable
  std::unique_ptr<num_samples_t[]> agg_nsamples;
  // assignment to variables, see variable.h for more detail
  std::unique_ptr<variable_value_t[]> assignments_free;
  std::unique_ptr<variable_value_t[]> assignments_evid;

  std::unique_ptr<weight_value_t[]> weight_values;  // array of weight values
  std::unique_ptr<bool[]> weights_isfixed;  // array of whether weight is fixed

  InferenceResult(const CompactFactorGraph &fg, const Weight weights[],
                  const CmdParser &opts);

  // copy constructor
  InferenceResult(const InferenceResult &other);

  void load_weights(const Weight weights[]);
  void load_assignments(std::ifstream &binary_input);
  void merge_weights_from(const InferenceResult &other);
  void average_regularize_weights(double current_stepsize);
  void copy_weights_to(InferenceResult &other) const;
  void show_weights_snippet(std::ostream &output) const;
  // XXX(igozali): Smells like an enum to me...
  void dump_weights_in_text(std::ostream &text_output) const;
  /*
   * This is actually the inverse of FactorGraph::dump_weights. It kinda makes
   * sense if you think about it...
   */
  void dump_weights_in_binary(std::ofstream &binary_output) const;
  void dump_assignments_in_binary(std::ofstream &binary_output) const;

  void clear_variabletally();
  void aggregate_marginals_from(const InferenceResult &other);
  void show_marginal_snippet(std::ostream &output) const;
  void show_marginal_histogram(std::ostream &output) const;
  void dump_marginals_in_text(std::ostream &text_output) const;

 private:
  InferenceResult(const CompactFactorGraph &fg, const CmdParser &opts);
};

}  // namespace dd

#endif  // DIMMWITTED_INFERENCE_RESULT_H_
