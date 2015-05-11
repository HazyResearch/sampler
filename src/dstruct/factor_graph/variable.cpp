#include "dstruct/factor_graph/variable.h"

namespace dd{

	Variable::Variable(){}

  Variable::Variable(const long & _id, const int & _domain_type, 
           const bool & _is_evid, const VariableValue & _lower_bound,
           const VariableValue & _upper_bound, const VariableValue & _init_value, 
           const VariableValue & _current_value, const int & _n_factors){

    this->id = _id;
    this->domain_type = _domain_type;
    this->is_evid = _is_evid;
    this->lower_bound = _lower_bound;
    this->upper_bound = _upper_bound;
    this->assignment_evid = _init_value;
    this->assignment_free = _current_value;

    this->n_factors = _n_factors;
    messages0 = NULL;
    messages1 = NULL;
  }

  void Variable::init_messages() {
    if (messages0 == NULL) {
      messages0 = new double[n_factors];
    }
    if (messages1 == NULL) {
      messages1 = new double[n_factors];
    }
    for (int i = 0; i < n_factors; i++) {
      messages0[i] = 0;
      messages1[i] = 0;
    }
  }

  bool VariableInFactor::satisfiedUsing(int value) const{
    return is_positive ? equal_to == value : !(equal_to == value); 
  }

  VariableInFactor::VariableInFactor(){

  }


  VariableInFactor::VariableInFactor(int dummy,
                    const int & _dimension, 
                    const long & _vid, const int & _n_position, 
                   const bool & _is_positive){
    this->dimension = _dimension;
    this->vid = _vid;
    this->n_position = _n_position;
    this->is_positive = _is_positive;
    this->equal_to = 1.0;
  }


  VariableInFactor::VariableInFactor(const long & _vid, const int & _n_position, 
                   const bool & _is_positive){
    this->dimension = -1;
    this->vid = _vid;
    this->n_position = _n_position;
    this->is_positive = _is_positive;
    this->equal_to = 1.0;
  }

  VariableInFactor::VariableInFactor(const long & _vid, const int & _n_position, 
                   const bool & _is_positive, const VariableValue & _equal_to){
    this->dimension = -1;
    this->vid = _vid;
    this->n_position = _n_position;
    this->is_positive = _is_positive;
    this->equal_to = _equal_to;
  }
}