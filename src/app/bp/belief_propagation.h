// class for belief propagation

#include <iostream>
#include <thread>
#include <vector>
#include "io/cmd_parser.h"
#include "dstruct/factor_graph/factor_graph.h"

#ifndef _BELIEF_PROPOGATION_H_
#define _BELIEF_PROPOGATION_H_

using namespace dd;

// bp task
void single_thread_bp_task(FactorGraph * const fg, int i_worker, int n_worker, bool vtof);

/**
 * Class for belief propagation
 * 
 * This class encapsulates bp learning and inference, and dumping results.
 */
class BeliefPropagation {
public:

  // factor graph
  FactorGraph * const fg;

  // command line parser
  CmdParser * const cmd_parser;

  // threads
  int n_threads;
  std::vector<std::thread> threads;

  // quiet output
  bool is_quiet;

  // marginals
  std::vector<double> marginals;

  /**
   * Constructs BeliefPropagation class with given factor graph, command line parser, etc.
   */
  BeliefPropagation(FactorGraph * const fg, CmdParser * const cmd_parser, int n_threads);

  /**
   * Performs inference
   * n_epoch number of epochs. A epoch is one pass over data
   */
  void inference(int n_epoch);
  void dump_inference_result();

  void init_messages();


};

#endif