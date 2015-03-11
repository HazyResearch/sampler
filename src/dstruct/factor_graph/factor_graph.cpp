
#include <iostream>
#include <memory>
#include "io/binary_parser.h"
#include "dstruct/factor_graph/factor_graph.h"
#include "dstruct/factor_graph/factor.h"
#include "timer.h"

dd::FactorGraph::FactorGraph(long _n_var, long _n_factor, long _n_weight, long _n_edge) : 
  n_var(_n_var), n_factor(_n_factor), n_weight(_n_weight), n_edge(_n_edge),
  c_nvar(0), c_nfactor(0), c_nweight(0), c_nedge(0), n_evid(0), n_query(0),
  variables(new Variable[_n_var]),
  factors(new Factor[_n_factor]),
  edges(new Edge[_n_edge]),
  weights(new Weight[_n_weight]),
  compact_factors(new CompactFactor[_n_edge]),
  compact_factors_weightids(new int[_n_edge]),
  factor_ids(new long[_n_edge]),
  vifs(new VariableInFactor[_n_edge]),
  infrs(new InferenceResult(_n_var, _n_weight)) {}

void dd::FactorGraph::copy_from(const FactorGraph * const p_other_fg){
  // copy each member from the given graph
  memcpy(variables, p_other_fg->variables, sizeof(Variable)*n_var);
  memcpy(factors, p_other_fg->factors, sizeof(Factor)*n_factor);
  memcpy(weights, p_other_fg->weights, sizeof(Weight)*n_weight);
  memcpy(factor_ids, p_other_fg->factor_ids, sizeof(long)*n_edge);
  memcpy(vifs, p_other_fg->vifs, sizeof(VariableInFactor)*n_edge);

  memcpy(compact_factors, p_other_fg->compact_factors, sizeof(CompactFactor)*n_edge);
  memcpy(compact_factors_weightids, p_other_fg->compact_factors_weightids, sizeof(int)*n_edge);

  c_nvar = p_other_fg->c_nvar;
  c_nfactor = p_other_fg->c_nfactor;
  c_nweight = p_other_fg->c_nweight;
  c_nedge = p_other_fg->c_nedge; 
  c_edge = p_other_fg->c_edge;

  infrs->init(variables, weights);
  infrs->ntallies = p_other_fg->infrs->ntallies;
  infrs->multinomial_tallies = new int[p_other_fg->infrs->ntallies];
  for(long i=0;i<infrs->ntallies;i++){
    infrs->multinomial_tallies[i] = p_other_fg->infrs->multinomial_tallies[i];
  }
}

long dd::FactorGraph::get_multinomial_weight_id(const VariableValue *assignments, const CompactFactor& fs, long vid, long proposal) {
  /**
   * The weight ids are aligned in a continuous region according
   * to the numerical order of variable values. 
   * Say for two variables v1, v2, v3, with cardinality d. The numerical value is
   * v1 * d^2 + v2 * d + v3.
   */
  long weight_offset = 0;
  // for each variable in the factor
  for (long i = fs.n_start_i_vif; i < fs.n_start_i_vif + fs.n_variables; i++) {
    const VariableInFactor & vif = vifs[i];
    if (vif.vid == vid) {
      weight_offset = weight_offset * (variables[vif.vid].upper_bound+1) + proposal;
    } else {
      weight_offset = weight_offset * (variables[vif.vid].upper_bound+1) + assignments[vif.vid];
    }
  }
  long base_offset = &fs - compact_factors; // note c++ will auto scale by sizeof(CompactFactor)
  return *(compact_factors_weightids + base_offset) + weight_offset;
}


void dd::FactorGraph::update_weight(const Variable & variable){
  // corresponding factors and weights in a continous region
  CompactFactor * const fs = compact_factors + variable.n_start_i_factors;
  const int * const ws = compact_factors_weightids + variable.n_start_i_factors;
  // for each factor
  for(long i=0;i<variable.n_factors;i++){
    // boolean variable
    if (variable.domain_type == DTYPE_BOOLEAN) {
      // only update weight when it is not fixed
      if(infrs->weights_isfixed[ws[i]] == false){
        // stochastic gradient ascent 
        // increment weight with stepsize * gradient of weight
        // gradient of weight = E[f|D] - E[f], where D is evidence variables, 
        // f is the factor function, E[] is expectation. Expectation is calculated
        // using a sample of the variable.
        infrs->weight_values[ws[i]] += 
          stepsize * (this->template potential<false>(fs[i]) - this->template potential<true>(fs[i]));
      }
    } else if (variable.domain_type == DTYPE_MULTINOMIAL) {
      // two weights need to be updated
      // sample with evidence fixed, I0, with corresponding weight w1
      // sample without evidence unfixed, I1, with corresponding weight w2 
      // gradient of wd0 = f(I0) - I(w1==w2)f(I1)
      // gradient of wd1 = I(w1==w2)f(I0) - f(I1)
      long wid1 = get_multinomial_weight_id(infrs->assignments_evid, fs[i], -1, -1);
      long wid2 = get_multinomial_weight_id(infrs->assignments_free, fs[i], -1, -1);
      int equal = (wid1 == wid2);

      if(infrs->weights_isfixed[wid1] == false){
        infrs->weight_values[wid1] += 
          stepsize * (this->template potential<false>(fs[i]) - equal * this->template potential<true>(fs[i]));
      }

      if(infrs->weights_isfixed[wid2] == false){
        infrs->weight_values[wid2] += 
          stepsize * (equal * this->template potential<false>(fs[i]) - this->template potential<true>(fs[i]));
      }
    }
  }
}

void dd::FactorGraph::load(const CmdParser & cmd, const bool is_quiet){

  // get factor graph file names from command line arguments
  std::string weight_file = cmd.weight_file->getValue();
  std::string variable_file = cmd.variable_file->getValue();
  std::string factor_file = cmd.factor_file->getValue();
  std::string edge_file = cmd.edge_file->getValue();

  std::string filename_edges = edge_file;
  std::string filename_factors = factor_file;
  std::string filename_variables = variable_file;
  std::string filename_weights = weight_file;

  Timer t;
  t.restart();
  double elapsed = 0;

  // load variables
  long long n_loaded = read_variables(filename_variables, *this);
  assert(n_loaded == n_var);
  if (!is_quiet) {
    std::cout << "LOADED VARIABLES: #" << n_loaded << std::endl;
    std::cout << "         N_QUERY: #" << n_query << std::endl;
    std::cout << "         N_EVID : #" << n_evid << std::endl;  
  }

  // load factors
  n_loaded = read_factors(filename_factors, *this);
  assert(n_loaded == n_factor);
  if (!is_quiet) {
    std::cout << "LOADED FACTORS: #" << n_loaded << std::endl;
  }

  // load weights
  n_loaded = read_weights(filename_weights, *this);
  assert(n_loaded == n_weight);
  if (!is_quiet) {
    std::cout << "LOADED WEIGHTS: #" << n_loaded << std::endl;
  }
  elapsed += t.elapsed();

  infrs->init(variables, weights);
  
  // load edges
  n_loaded = read_edges(edge_file, *this);
  if (!is_quiet) {
    std::cout << "LOADED EDGES: #" << n_loaded << std::endl;
  }
  
  // construct edge-based store
  this->organize_graph_by_edge();
  this->safety_check();

  elapsed = t.elapsed();
  if (!is_quiet) {
    std::cout << "LOADING TIME: " << elapsed << std::endl;
  }

}

bool dd::compare_position(const VariableInFactor& x, const VariableInFactor& y) {
  return x.n_position - y.n_position;
}

void dd::FactorGraph::organize_graph_by_edge() {
  // number of edges
  c_edge = 0;

  // use compact adjacency list to build factor graph
  // prev_vid and prev_fid are used to store the nearest previous edge with
  // the same variable id or factor id in the edge list
  // last_vid and last_fid are used to store the last edge index in the edge
  // list, and the index in last_vid or last_fid is corresponding to the
  // variable id or factor id
  std::unique_ptr<long[]> prev_vid(new long[n_edge]);
  std::unique_ptr<long[]> last_vid(new long[n_edge]);
  std::unique_ptr<long[]> prev_fid(new long[n_edge]);
  std::unique_ptr<long[]> last_fid(new long[n_edge]);

  // initial last_vid, -1 represents null
  for (long i = 0; i < n_var; i++) {
    last_vid[i] = -1;
  }

  // initial last_fid, -1 represents null
  for (long i = 0; i < n_factor; i++) {
    last_fid[i] = -1;
  }

  // build compact adjacency list 
  for (long i = 0; i < n_edge; i++) {
    Edge & edge = edges[i];
    prev_vid[i] = last_vid[edge.variable_id];
    last_vid[edge.variable_id] = i;
    prev_fid[i] = last_fid[edge.factor_id];
    last_fid[edge.factor_id] = i;
  }

  for (long i = 0; i < n_factor; i++) {
    Factor & factor = factors[i];
    factor.n_start_i_vif = c_edge;
    long start_idx = c_edge;
    // find all edges corresponding to the factor, cur is index of last one 
    // in edge compact adjacency list
    long cur = last_fid[factor.id];
    while (cur >= 0) {
      Edge & edge = edges[cur];
      if (variables[edge.variable_id].domain_type == DTYPE_BOOLEAN) {
          vifs[c_edge] = dd::VariableInFactor(edge.variable_id, variables[edge.variable_id].upper_bound, edge.variable_id, edge.position, edge.ispositive);
      } else {
          vifs[c_edge] = dd::VariableInFactor(edge.variable_id, edge.position, edge.ispositive, edge.equal_predicate);
      }
      c_edge++;
      // use prev_fid to find the previous edge in the list
      cur = prev_fid[cur];
    }
    // sort variables in factor by position in factor
    std::sort(vifs + start_idx, vifs + c_edge, dd::compare_position);
  }
  
  c_edge = 0;
  long ntallies = 0;
  // for each variable, put the factors into factor_dups
  for(long i=0;i<n_var;i++){
    Variable & variable = variables[i];
    
    variable.n_start_i_factors = c_edge;
    if(variable.domain_type == DTYPE_MULTINOMIAL){
      variable.n_start_i_tally = ntallies;
      ntallies += variable.upper_bound - variable.lower_bound + 1;
    }
    long cnt = 0;
    // find all edges corresponding to the variable, cur is index of last one
    // in edge compact adjacency list    
    long cur = last_vid[variable.id];
    while (cur >= 0) {
      long long & fid = edges[cur].factor_id;
      factor_ids[c_edge] = fid;
      compact_factors[c_edge].id = factors[fid].id;
      compact_factors[c_edge].func_id = factors[fid].func_id;
      compact_factors[c_edge].n_variables = factors[fid].n_variables;
      compact_factors[c_edge].n_start_i_vif = factors[fid].n_start_i_vif;
      compact_factors_weightids[c_edge] = factors[fid].weight_id;
      cnt ++;
      c_edge ++;
      // use prev_fid to find the previous edge in the list
      cur = prev_vid[cur];
    }
    variable.n_factors = cnt;
  }
}

void dd::FactorGraph::safety_check(){

  // check whether variables, factors, and weights are stored 
  // in the order of their id
  long s = n_var;
  for(long i=0;i<s;i++){
    assert(this->variables[i].id == i);
  }
  s = n_factor;
  for(long i=0;i<s;i++){
    assert(this->factors[i].id == i);
  }
  s = n_weight;
  for(long i=0;i<s;i++){
    assert(this->weights[i].id == i);
  }
}






