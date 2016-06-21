#ifndef DIMMWITTED_DIMMWITTED_H_
#define DIMMWITTED_DIMMWITTED_H_

#include "cmd_parser.h"
#include "factor_graph.h"
#include "gibbs_sampler.h"
#include <memory>

namespace dd {

/**
 * Command-line interface.
 */
int dw(int argc, const char* const argv[]);

/**
 * Runs gibbs sampling using the given command line parser
 */
int gibbs(const CmdParser& opts);

int init(const CmdParser& cmd_parser);
int learn(const CmdParser& cmd_parser);
int inference(const CmdParser& cmd_parser);

/**
 * Class for (NUMA-aware) gibbs sampling
 *
 * This class encapsulates gibbs learning and inference, and dumping results.
 * Note the factor graph is copied on each NUMA node.
 */
class DimmWitted {
 public:
  const Weight* const weights;  // TODO clarify ownership

  // command line parser
  const CmdParser& opts;  // TODO clarify ownership

  // the highest node number available
  // actually, number of NUMA nodes = n_numa_nodes + 1
  size_t n_numa_nodes;

  // number of threads per NUMA node
  size_t n_thread_per_numa;

  // factor graph copies per NUMA node
  std::vector<GibbsSampler> samplers;

  /**
   * Constructs DimmWitted class with given factor graph, command line
   * parser,
   * and number of data copies. Allocate factor graph to NUMA nodes.
   * n_datacopy number of factor graph copies. n_datacopy = 1 means only
   * keeping one factor graph.
   */
  DimmWitted(std::unique_ptr<CompactFactorGraph> p_cfg, const Weight weights[],
             const CmdParser& opts);

  DimmWitted(const Weight weights[], const CmdParser& opts);

  /**
   * Performs learning
   * n_epoch number of epochs. A epoch is one pass over data
   * n_sample_per_epoch not used any more.
   * stepsize starting step size for weight update (aka learning rate)
   * decay after each epoch, the stepsize is updated as stepsize = stepsize *
   * decay
   * reg_param regularization parameter
   * is_quiet whether to compress information display
   */
  void learn();

  /**
   * Performs inference
   * n_epoch number of epochs. A epoch is one pass over data
   * is_quiet whether to compress information display
   */
  void inference();

  /**
   * Aggregates results from different NUMA nodes
   * Dumps the inference result for variables
   * is_quiet whether to compress information display
   */
  void aggregate_results_and_dump();

  /**
   * Dumps the learned weights
   * is_quiet whether to compress information display
   */
  void dump_weights();

  /**
   */
  void load_weights();

  /**
   * Checkpoints the learned weights, and the assignments in each of the sampler.
   */
  void checkpoint(bool should_include_weights);

  /**
   * Loads the checkpointed weights and assignments in each node.
   */
  void resume();

 private:
  num_epochs_t compute_n_epochs(num_epochs_t n_epoch);
};

}  // namespace dd

#endif  // DIMMWITTED_DIMMWITTED_H_
