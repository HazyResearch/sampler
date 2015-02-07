#include "dstruct/factor_graph/inference_result.h"
#include <iostream>

dd::InferenceResult::InferenceResult(long _nvars, long _nweights):
  nvars(_nvars),
  nweights(_nweights),
  agg_means(new double[_nvars]),
  agg_nsamples(new double[_nvars]),
  assignments_free(new VariableValue[_nvars]),
  assignments_evid(new VariableValue[_nvars]),
  weight_values(new double [_nweights]),
  weights_isfixed(new bool [_nweights]) {}

dd::InferenceResult::~InferenceResult() {
  std::cerr << "destroying inference_result..." << std::endl;
  delete[] agg_nsamples;
  delete[] agg_means;
  delete[] assignments_free;
  delete[] assignments_evid;
  delete[] weight_values;
  delete[] weights_isfixed;
}

void dd::InferenceResult::init(Variable * variables, Weight * weights){
  init_variables(variables);
  init_weights(weights);
}

void dd::InferenceResult::init_weights(const Weight * weights) {
  for(long t=0;t<nweights;t++){
    const Weight & weight = weights[t];
    weight_values[weight.id] = weight.weight;
    weights_isfixed[weight.id] = weight.isfixed;
  }
}

void dd::InferenceResult::init_variables(const Variable * variables) {
  ntallies = 0;
  for(long t=0;t<nvars;t++){
    const Variable & variable = variables[t];
    assignments_free[variable.id] = variable.assignment_free;
    assignments_evid[variable.id] = variable.assignment_evid;
    agg_means[variable.id] = 0.0;
    agg_nsamples[variable.id] = 0.0;
    if(variable.domain_type == DTYPE_MULTINOMIAL){
      ntallies += variable.upper_bound - variable.lower_bound + 1;
    }
  }

  multinomial_tallies = new int[ntallies];
  for(long i=0;i<ntallies;i++){
    multinomial_tallies[i] = 0;
  }
}

void dd::InferenceResult::init_variables(const Variable * variables, long num_variables,
  long idoffset) {
  long pos = 0;
  std::cerr << "nvars " << nvars << std::endl;
  std::cerr << "idoffset " << idoffset << std::endl;
  for (long t = 0; t < num_variables; t++) {
    const Variable & variable = variables[t];
    pos = variable.id + idoffset;
    // std::cerr << "pos " << pos << std::endl;
    assignments_free[pos] = variable.assignment_free;
    assignments_evid[pos] = variable.assignment_evid;
    agg_means[pos] = 0.0;
    agg_nsamples[pos] = 0.0;
  }
}

void dd::InferenceResult::init_tallies(long _ntallies) {
  ntallies = _ntallies;
  multinomial_tallies = new int[ntallies];
  for(long i=0;i<ntallies;i++){
    multinomial_tallies[i] = 0;
  }
}