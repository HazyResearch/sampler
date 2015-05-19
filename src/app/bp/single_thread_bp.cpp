/************************************
 * Single thread belief propagation
 * NOTE messages are in log domain
 ************************************/
#include <cmath>
#include <vector>
#include "app/bp/single_thread_bp.h"
#include "common.h"

SingleThreadBP::SingleThreadBP(FactorGraph * fg) : fg(fg) {}

void SingleThreadBP::send_message_block(int i_sharding, int n_sharding, bool vtof) {
  long num   = vtof ? fg->n_var : fg->n_factor;
  // calculates the start and end id in this partition
  long start = (num / n_sharding + 1) * i_sharding;
  long end   = (num / n_sharding + 1) * (i_sharding + 1);
  end = end > num ? num : end;
  for (long i = start; i < end; i++) {
    vtof ? send_message_from_variable(i) : send_message_from_factor(i);
  }
}

void SingleThreadBP::send_message_from_variable(long vid) {
  Variable &variable = fg->variables[vid];
  for (long fid : variable.tmp_factor_ids) {
    send_message_vtof(vid, fid);
  }
}

void SingleThreadBP::send_message_from_factor(long fid) {
  Factor &factor = fg->factors[fid];
  for (VariableInFactor &v : factor.tmp_variables) {
    send_message_ftov(fid, v.vid);
  }
}

void SingleThreadBP::normalize(double messages[]) {
  double sum = logadd(messages[0], messages[1]);
  for (int i = 0; i < 2; i++) {
    messages[i] -= sum;
  }
}

void SingleThreadBP::send_message_vtof(long vid, long fid) {
  if (DEBUG && vid == 0) printf("\nsend_message_vtof %ld -> %ld\n", vid, fid);
  Variable &variable = fg->variables[vid];
  Factor &factor     = fg->factors[fid]; 
  // collect messages from other factors
  // only support boolean
  double messages[2] = {0, 0};
  for (int i = 0; i < variable.tmp_factor_ids.size(); i++) {
    long fid_other = variable.tmp_factor_ids[i];
    if (fid_other == fid) continue;
    messages[0] += variable.messages0[i];
    messages[1] += variable.messages1[i];
  }

  // normalize messages
  normalize(messages);

  // send message to factor
  int i;
  for (i = 0; i < factor.tmp_variables.size(); i++) {
    if (factor.tmp_variables[i].vid == vid) {
      break;
    }
  }

  if (DEBUG && vid == 0) printf("message0 = %f, message1 = %f\n", messages[0], messages[1]);

  factor.messages0[i] = messages[0];
  factor.messages1[i] = messages[1];
}

void SingleThreadBP::send_message_ftov(long fid, long vid) {
  if (DEBUG && vid == 0) printf("\nsend_message_ftov %ld -> %ld\n", fid, vid);
  Variable &variable = fg->variables[vid];
  Factor &factor     = fg->factors[fid];
  
  // marginalizing other variables
  // note this is for generating possible worlds for boolean
  int num_states = (int)pow(2, factor.n_variables - 1);
  double messages[2] = {MINUS_INFINITY, MINUS_INFINITY};

  for (int state = 0; state < num_states; state++) {
    // set up variable assignment
    int j = 0;
    for (VariableInFactor &v : factor.tmp_variables) {
      long cur_vid = v.vid;
      if (cur_vid == vid) continue;
      fg->infrs->assignments_free[cur_vid] = state & (1 << j);
      j++;
    }

    double inner_messages[2] = {0, 0};
    // for each value in domain
    for (int j = 0; j < 2; j++) {
      // compute weighted factor
      double weighted_factor = fg->infrs->weight_values[factor.weight_id]
        * factor.potential(fg->vifs, fg->infrs->assignments_free, vid, j);
      if (DEBUG && vid == 0) {
        printf("state = %d, current value = %d\n", state, j);
        printf("wid = %ld weight = %f weighted_factor = %f\n", 
          factor.weight_id, fg->infrs->weight_values[factor.weight_id], weighted_factor);
      }

      // collect messages from other variables
      for (int i = 0; i < factor.tmp_variables.size(); i++) {
        long vid_other = factor.tmp_variables[i].vid;
        if (vid_other == vid) continue;
        if (fg->infrs->assignments_free[vid_other] == 0) {
          inner_messages[j] += factor.messages0[i];
        } else {
          inner_messages[j] += factor.messages1[i];
        }
      }
      if (DEBUG && vid == 0) {
        printf("inner_message = %f\n", inner_messages[j]);
      }
      inner_messages[j] += weighted_factor;
      messages[j] = logadd(messages[j], inner_messages[j]);
    }
  }

  // normalize messages
  normalize(messages);

  // send message to variable
  int i;
  for (i = 0; i < variable.tmp_factor_ids.size(); i++) {
    if (variable.tmp_factor_ids[i] == fid) {
      break;
    }
  }
  variable.messages0[i] = messages[0];
  variable.messages1[i] = messages[1];
  if (DEBUG && vid == 0) printf("message0 = %f, message1 = %f\n", messages[0], messages[1]);
}