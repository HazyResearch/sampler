
#include "app/gibbs/single_thread_sampler.h"
#include <stdlib.h>
#include <thread>
#include "common.h"
#include "message.h"

#ifndef _SINGLE_NODE_SAMPLER_H
#define _SINGLE_NODE_SAMPLER_H

namespace dd{

  /**
   * Class for a single NUMA node sampler
   */
  class SingleNodeSampler{

  public:
    // factor graph
    FactorGraph * const p_fg;
    // number of threads
    int nthread;
    // node id
    int nodeid;

    bool sample_evidence;
    int burn_in;
    bool learn_non_evidence;

    std::vector<std::thread> threads;

    /**
     * Constructs a SingleNodeSampler given factor graph, number of threads, and
     * node id.
     */
    SingleNodeSampler(FactorGraph * _p_fg, int _nthread, int _nodeid);
    SingleNodeSampler(FactorGraph * _p_fg, int _nthread, int _nodeid, 
      bool sample_evidence, int burn_in);
    SingleNodeSampler(FactorGraph * _p_fg, int _nthread, int _nodeid, 
      bool sample_evidence, int burn_in, bool learn_non_evidence);

    /**
     * Clears the inference results in this sampler
     */
    void clear_variabletally();

    /**
     * Performs sample
     */
    void sample(int i_epoch);

    void sample_inc();

    /**
     * Waits for sample worker to finish
     */
    void wait();

    /**
     * Performs SGD
     */
    void sample_sgd();

    /**
     * Waits for sgd worker to finish
     */
    void wait_sgd();

    void learn_fusion(FusionMessage * msg);
    void save_fusion_message(FusionMessage * msg);

  };
}

#endif






