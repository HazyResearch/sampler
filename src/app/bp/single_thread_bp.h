// class for single thread belief propagation
#include "dstruct/factor_graph/factor_graph.h"
#include "timer.h"

#ifndef _SINGLE_THREAD_BP_
#define _SINGLE_THREAD_BP_

using namespace dd;

class SingleThreadBP {
public:
  // factor graph
  FactorGraph * const fg;

  /**
   * Constructs a SingleThreadBP with given factor graph
   */ 
  SingleThreadBP(FactorGraph * fg);

  /**
   * Send messages. The variables are divided into n_sharding equal partitions
   * based on their ids. This function samples variables in the i_sharding-th 
   * partition.
   */ 
  void send_message_block(int i_sharding, int n_sharding, bool vtof);

  /**
   * send message from a single variable/factor to all its neighbors
   */
  void send_message_from_variable(long vid);
  void send_message_from_factor(long fid);

  // send message from from the given variable/factor to the given factor/variable
  void send_message_vtof(long vid, long fid);
  void send_message_ftov(long fid, long vid);

private:
  void normalize(double messages[]);
};

#endif