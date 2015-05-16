
#include "factor.h"
#include <cstddef>

namespace dd{

  CompactFactor::CompactFactor(){}

  CompactFactor::CompactFactor(const long & _id){
    id = _id;
  }

  Factor::Factor(){

  }

  Factor::Factor(const FactorIndex & _id,
         const WeightIndex & _weight_id,
         const int & _func_id,
         const int & _n_variables){
    this->id = _id;
    this->weight_id = _weight_id;
    this->func_id = _func_id;
    this->n_variables = _n_variables;
    messages0 = NULL;
    messages1 = NULL;
  }

  void Factor::init_messages() {
    if (messages0 == NULL) {
      messages0 = new double[n_variables];
    }
    if (messages1 == NULL) {
      messages1 = new double[n_variables];
    }
    for (int i = 0; i < n_variables; i++) {
      messages0[i] = 0;
      messages1[i] = 0;
    }
  }
 
}




