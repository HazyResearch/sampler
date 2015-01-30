  
#include "dstruct/factor_graph/variable.h"
#include "dstruct/factor_graph/weight.h"

#ifndef _INFERENCE_RESULT_H_
#define _INFERENCE_RESULT_H_

namespace dd {
  /** 
   * Encapsulates inference result statistics
   */
  class InferenceResult{
  public:

    long nvars;     // number of variables
    long nweights;  // number of weights
    long ntallies;

    int * multinomial_tallies; // this might be slow...

    // array of sum of samples for each variable
    double * agg_means; 
    // array of number of samples for each variable
    double * agg_nsamples; 
    // assignment to variables, see variable.h for more detail
    VariableValue * assignments_free;
    VariableValue * assignments_evid;
    double * const weight_values; // array of weight values
    bool * const weights_isfixed; // array of whether weight is fixed

    InferenceResult(long _nvars, long _nweights);
    ~InferenceResult();

    /**
     * Initialize the class with given variables and weights
     */
    void init(Variable * variables, Weight * weights);
    void init_variables(const Variable * variables);
    void init_weights(const Weight * weights);
    void init_tallies(long _ntallies);
    void init_variables(const Variable * variables, long num_variables, long idoffset);
  };
}

#endif