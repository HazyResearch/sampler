#include "dimmwitted.h"
#include "assert.h"
#include "bin2text.h"
#include "binary_format.h"
#include "common.h"
#include "factor_graph.h"
#include "gibbs_sampler.h"
#include "text2bin.h"

#include <fstream>
#include <iomanip>
#include <map>
#include <unistd.h>

namespace dd {

// the command-line entry point
int dw(int argc, const char *const argv[]) {
  // available modes
  const std::map<std::string, int (*)(const CmdParser &)> MODES = {
      {"gibbs", gibbs},  // to do the learning and inference with Gibbs sampling
      {"init", init},  // to instantiate compact factor graph and weights
      {"learn", learn},  // to perform learning with Gibbs sampling
      {"inference", inference},  // to perform inference with Gibbs sampling
      {"text2bin", text2bin},  // to generate binary factor graphs from TSV
      {"bin2text", bin2text},  // to dump TSV of binary factor graphs
  };

  // parse command-line arguments
  CmdParser cmd_parser(argc, argv);

  // dispatch to the correct function
  const auto &mode = MODES.find(cmd_parser.app_name);
  return (mode != MODES.end()) ? mode->second(cmd_parser) : 1;
}

int init(const dd::CmdParser &opts) {
  // number of NUMA nodes
  size_t n_numa_node = numa_max_node() + 1;
  // number of max threads per NUMA node
  size_t n_thread_per_numa = (sysconf(_SC_NPROCESSORS_CONF)) / (n_numa_node);

  if (!opts.should_be_quiet) {
    std::cout << std::endl;
    std::cout << "#################MACHINE CONFIG#################"
              << std::endl;
    std::cout << "# # NUMA Node        : " << n_numa_node << std::endl;
    std::cout << "# # Thread/NUMA Node : " << n_thread_per_numa << std::endl;
    std::cout << "################################################"
              << std::endl;
    std::cout << std::endl;
    std::cout << opts << std::endl;
  }

  FactorGraphDescriptor meta = read_meta(opts.fg_file);
  if (!opts.should_be_quiet) {
    std::cout << meta << std::endl;
  }

  // Run on NUMA node 0
  numa_run_on_node(0);
  numa_set_localalloc();

  // Load factor graph
  dprintf("Initializing factor graph...\n");
  std::unique_ptr<FactorGraph> fg(new FactorGraph(meta));

  fg->load_variables(opts.variable_file);
  fg->load_weights(opts.weight_file);
  fg->load_domains(opts.domain_file);
  fg->load_factors(opts.factor_file);
  fg->safety_check();

  if (!opts.should_be_quiet) {
    std::cout << "Printing FactorGraph statistics:" << std::endl;
    std::cout << *fg << std::endl;
  }

  std::unique_ptr<CompactFactorGraph> cfg(new CompactFactorGraph(*fg));
  cfg->dump(opts.snapshot_path);

  // Initialize Gibbs sampling application.
  DimmWitted dw(std::move(cfg),
      fg->weights.get(), opts);

  // Explicitly drop the raw factor graph used only during loading
  fg.release();

  // In the very beginning, checkpointing the weights is not required since
  // the weights should be equal to the one defined in the weights file.
  dw.checkpoint(false);

  return 0;
}

int learn(const dd::CmdParser &opts) {
  /*
   * This will be some awkward shit I'll ever write, but let's see what Jaeho
   * thinks. I'm going to create a factor graph to load the weights array, but
   * only for that purpose. At least until Weight[] ownership is clarified.
   */
  std::cout << "Performing learning on persisted compact factor graph...";
  FactorGraphDescriptor meta = read_meta(opts.fg_file);
  std::unique_ptr<FactorGraph> fg(new FactorGraph(meta));

  fg->load_weights(opts.weight_file);

  std::unique_ptr<CompactFactorGraph> cfg(new CompactFactorGraph(meta));
  cfg->resume(opts.snapshot_path);

  DimmWitted dw(std::move(cfg), fg->weights.get(), opts);

  fg.release();

  // Restore the factor graph assignments and weights.
  dw.resume();

  // learning
  dw.learn();

  // Similar to the main `gibbs` workflow, dump the weights as well.
  dw.dump_weights();

  // Checkpoint the state, but this time, checkpoint the weights as well.
  dw.checkpoint(true);

  return 0;
}

int inference(const dd::CmdParser &opts) {
  /*
   * This will be some awkward shit I'll ever write, but let's see what Jaeho
   * thinks. I'm going to create a factor graph to load the weights array, but
   * only for that purpose. At least until Weight[] ownership is clarified.
   */
  std::cout << "Performing inference on persisted compact factor graph...";
  FactorGraphDescriptor meta = read_meta(opts.fg_file);
  std::unique_ptr<FactorGraph> fg(new FactorGraph(meta));

  fg->load_weights(opts.weight_file);

  std::unique_ptr<CompactFactorGraph> cfg(new CompactFactorGraph(meta));
  cfg->resume(opts.snapshot_path);

  DimmWitted dw(std::move(cfg), fg->weights.get(), opts);

  fg.release();

  // Restore the factor graph assignments and weights.
  dw.resume();

  // inference
  dw.inference();
  dw.aggregate_results_and_dump();

  return 0;
}

int gibbs(const dd::CmdParser &args) {
  // number of NUMA nodes
  size_t n_numa_node = numa_max_node() + 1;
  // number of max threads per NUMA node
  size_t n_thread_per_numa = (sysconf(_SC_NPROCESSORS_CONF)) / (n_numa_node);

  if (!args.should_be_quiet) {
    std::cout << std::endl;
    std::cout << "#################MACHINE CONFIG#################"
              << std::endl;
    std::cout << "# # NUMA Node        : " << n_numa_node << std::endl;
    std::cout << "# # Thread/NUMA Node : " << n_thread_per_numa << std::endl;
    std::cout << "################################################"
              << std::endl;
    std::cout << std::endl;
    std::cout << args << std::endl;
  }

  FactorGraphDescriptor meta = read_meta(args.fg_file);
  std::cout << "Factor graph to load:\t" << meta << std::endl;

  // Run on NUMA node 0
  numa_run_on_node(0);
  numa_set_localalloc();

  // Load factor graph
  dprintf("Initializing factor graph...\n");
  std::unique_ptr<FactorGraph> fg(new FactorGraph(meta));

  fg->load_variables(args.variable_file);
  fg->load_weights(args.weight_file);
  fg->load_domains(args.domain_file);
  fg->load_factors(args.factor_file);
  std::cout << "Factor graph loaded:\t" << fg->size << std::endl;
  fg->safety_check();

  if (!args.should_be_quiet) {
    std::cout << "Printing FactorGraph statistics:" << std::endl;
    std::cout << *fg << std::endl;
  }

  // Initialize Gibbs sampling application.
  DimmWitted dw(
      std::unique_ptr<CompactFactorGraph>(new CompactFactorGraph(*fg)),
      fg->weights.get(), args);
  std::cout << "I'm here debugging again" << std::endl;

  // Explicitly drop the raw factor graph used only during loading
  fg.release();

  // learning
  dw.learn();

  // dump weights
  dw.dump_weights();

  // inference
  dw.inference();
  dw.aggregate_results_and_dump();

  return 0;
}

DimmWitted::DimmWitted(std::unique_ptr<CompactFactorGraph> p_cfg,
                       const Weight weights[], const CmdParser &opts)
    : weights(weights), opts(opts) {
  n_numa_nodes = numa_max_node() + 1;
  if (opts.n_datacopy > 0 && opts.n_datacopy < n_numa_nodes) {
    n_numa_nodes = opts.n_datacopy;
  }

  // limit number of NUMA nodes such that each gets at least one thread
  if (opts.n_threads < n_numa_nodes) n_numa_nodes = opts.n_threads;

  // max possible threads per NUMA node
  n_thread_per_numa = opts.n_threads / n_numa_nodes;

  // copy factor graphs
  for (size_t i = 0; i < n_numa_nodes; ++i) {
    numa_run_on_node(i);
    numa_set_localalloc();

    std::cout << "CREATE CFG ON NODE ..." << i << std::endl;
    samplers.push_back(GibbsSampler(
        std::unique_ptr<CompactFactorGraph>(
            i == 0 ?
                   // use the given factor graph for the first sampler
                p_cfg.release()
                   :
                   // then, make a copy for the rest
                new CompactFactorGraph(samplers[0].fg)),
        weights, n_thread_per_numa, i, opts));
  }
}

void DimmWitted::inference() {
  const num_epochs_t n_epoch = compute_n_epochs(opts.n_inference_epoch);
  const variable_id_t nvar = samplers[0].fg.size.num_variables;
  const bool should_show_progress = !opts.should_be_quiet;
  Timer t_total, t;

  for (auto &sampler : samplers) sampler.infrs.clear_variabletally();

  // inference epochs
  for (num_epochs_t i_epoch = 0; i_epoch < n_epoch; ++i_epoch) {
    if (should_show_progress) {
      std::cout << std::setprecision(2) << "INFERENCE EPOCH "
                << i_epoch * n_numa_nodes << "~"
                << ((i_epoch + 1) * n_numa_nodes) << "...." << std::flush;
    }

    // restart timer
    t.restart();

    // sample
    for (auto &sampler : samplers) sampler.sample(i_epoch);

    // wait for samplers to finish
    for (auto &sampler : samplers) sampler.wait();

    double elapsed = t.elapsed();
    if (should_show_progress) {
      std::cout << "" << elapsed << " sec.";
      std::cout << "," << (nvar * n_numa_nodes) / elapsed << " vars/sec"
                << std::endl;
    }
  }

  double elapsed = t_total.elapsed();
  std::cout << "TOTAL INFERENCE TIME: " << elapsed << " sec." << std::endl;
}

void DimmWitted::learn() {
  InferenceResult &infrs = samplers[0].infrs;

  const num_epochs_t n_epoch = compute_n_epochs(opts.n_learning_epoch);
  const variable_id_t nvar = infrs.nvars;
  const weight_id_t nweight = infrs.nweights;
  const double decay = opts.decay;
  const bool should_show_progress = !opts.should_be_quiet;
  Timer t_total, t;

  double current_stepsize = opts.stepsize;
  const std::unique_ptr<double[]> prev_weights(new double[nweight]);
  COPY_ARRAY(infrs.weight_values.get(), nweight, prev_weights.get());

  // learning epochs
  for (num_epochs_t i_epoch = 0; i_epoch < n_epoch; ++i_epoch) {
    if (should_show_progress) {
      std::cout << std::setprecision(2) << "LEARNING EPOCH "
                << i_epoch * n_numa_nodes << "~"
                << ((i_epoch + 1) * n_numa_nodes) << "...." << std::flush;
    }

    t.restart();

    // performs stochastic gradient descent with sampling
    for (auto &sampler : samplers) sampler.sample_sgd(current_stepsize);

    // wait the samplers to finish
    for (auto &sampler : samplers) sampler.wait();

    // sum the weights and store in the first factor graph
    // the average weights will be calculated and assigned to all factor graphs
    for (size_t i = 1; i < n_numa_nodes; ++i)
      infrs.merge_weights_from(samplers[i].infrs);
    infrs.average_regularize_weights(current_stepsize);
    for (size_t i = 1; i < n_numa_nodes; ++i)
      infrs.copy_weights_to(samplers[i].infrs);

    // calculate the norms of the difference of weights from the current epoch
    // and last epoch
    double lmax = -INFINITY;
    double l2 = 0.0;
    for (weight_id_t j = 0; j < nweight; ++j) {
      double diff = fabs(prev_weights[j] - infrs.weight_values[j]);
      prev_weights[j] = infrs.weight_values[j];
      l2 += diff * diff;
      if (lmax < diff) lmax = diff;
    }
    lmax /= current_stepsize;

    double elapsed = t.elapsed();
    if (should_show_progress) {
      std::cout << "" << elapsed << " sec.";
      std::cout << "," << (nvar * n_numa_nodes) / elapsed << " vars/sec."
                << ",stepsize=" << current_stepsize << ",lmax=" << lmax
                << ",l2=" << sqrt(l2) / current_stepsize << std::endl;
    }

    current_stepsize *= decay;
  }

  double elapsed = t_total.elapsed();
  std::cout << "TOTAL LEARNING TIME: " << elapsed << " sec." << std::endl;
}

void DimmWitted::dump_weights() {
  // learning weights snippets
  const InferenceResult &infrs = samplers[0].infrs;

  if (!opts.should_be_quiet) infrs.show_weights_snippet(std::cout);

  // dump learned weights
  std::string filename_text(opts.output_folder +
                            "/inference_result.out.weights.text");
  std::cout << "DUMPING... TEXT    : " << filename_text << std::endl;
  std::ofstream fout_text(filename_text);
  infrs.dump_weights_in_text(fout_text);
  fout_text.close();
}

void DimmWitted::checkpoint(bool should_include_weights) {
  // Weights should be persisted after learning.
  if (should_include_weights) {
    std::ofstream weights_binary_out;
    weights_binary_out.open(opts.weight_file + ".out");

    // First sampler holds the merged weights after learning.
    //
    // Ideally, this should be LearningResult to emphasize the fact
    // that we're calling checkpoint only after learning is called.
    samplers[0].infrs.dump_weights_in_binary(weights_binary_out);
  }

  for (size_t i = 0; i < n_numa_nodes; ++i) {
    std::ofstream assignments_binary_out;
    assignments_binary_out.open(opts.snapshot_path + "/graph.assignments.out.part-" + std::to_string(i), std::ios::binary | std::ios::out);

    samplers[i].infrs.dump_assignments_in_binary(assignments_binary_out);
  }

  return;
}

void DimmWitted::resume() {
  for (size_t i = 0; i < n_numa_nodes; ++i) {
    samplers[i].infrs.load_weights(weights);

    std::ifstream assignments_binary_in;
    assignments_binary_in.open(opts.snapshot_path + "/graph.assignments.out.part-" + std::to_string(i), std::ios::binary | std::ios::in);

    samplers[i].infrs.load_assignments(assignments_binary_in);
  }
}

void DimmWitted::aggregate_results_and_dump() {
  InferenceResult &infrs = samplers[0].infrs;

  // aggregate assignments across all possible worlds
  for (size_t i = 1; i < n_numa_nodes; ++i)
    infrs.aggregate_marginals_from(samplers[i].infrs);

  if (!opts.should_be_quiet) infrs.show_marginal_snippet(std::cout);

  // dump inference results
  std::string filename_text(opts.output_folder + "/inference_result.out.text");
  std::cout << "DUMPING... TEXT    : " << filename_text << std::endl;
  std::ofstream fout_text(filename_text);
  infrs.dump_marginals_in_text(fout_text);
  fout_text.close();

  if (!opts.should_be_quiet) infrs.show_marginal_histogram(std::cout);
}

// compute number of NUMA-aware epochs for learning or inference
num_epochs_t DimmWitted::compute_n_epochs(num_epochs_t n_epoch) {
  return std::ceil((double)n_epoch / n_numa_nodes);
}

}  // namespace dd
