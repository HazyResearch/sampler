/**
 * This file contains binary formatting methods for FactorGraphs and
 * CompactFactorGraphs. We think this file is a good place to put the
 * definitions of these methods as it conveys the intentions quite clearly
 * (i.e. for binary formatting of these objects).
 */
#include "binary_format.h"
#include "common.h"
#include "factor.h"
#include "factor_graph.h"
#include "variable.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace dd {

// Read meta data file, return Meta struct
FactorGraphDescriptor read_meta(const std::string &meta_file) {
  FactorGraphDescriptor meta;
  std::ifstream file(meta_file);
  std::string buf;
  getline(file, buf, ',');
  meta.num_weights = atoll(buf.c_str());
  getline(file, buf, ',');
  meta.num_variables = atoll(buf.c_str());
  getline(file, buf, ',');
  meta.num_factors = atoll(buf.c_str());
  getline(file, buf, ',');
  meta.num_edges = atoll(buf.c_str());
  return meta;
}

void FactorGraph::load_weights(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  num_weights_t count = 0;
  while (file && file.peek() != EOF) {
    // read fields
    weight_id_t wid;
    uint8_t isfixed;
    weight_value_t initial_value;
    read_be_or_die(file, wid);
    read_be_or_die(file, isfixed);
    read_be_or_die(file, initial_value);
    // load into factor graph
    weights[wid] = Weight(wid, initial_value, isfixed);
    ++count;
  }
  size.num_weights += count;
}

void FactorGraph::load_variables(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  num_variables_t count = 0;
  while (file && file.peek() != EOF) {
    variable_id_t vid;
    uint8_t role_serialized;
    variable_value_t initial_value;
    uint16_t
        dtype_serialized;  // TODO shouldn't this come before the actual value?
    num_variable_values_t cardinality;
    // read fields
    read_be_or_die(file, vid);
    read_be_or_die(file, role_serialized);
    read_be_or_die(file, initial_value);
    read_be_or_die(file, dtype_serialized);
    read_be_or_die(file, cardinality);
    // map serialized to internal values
    ++count;
    variable_domain_type_t dtype;
    switch (dtype_serialized) {
      case 0:
        dtype = DTYPE_BOOLEAN;
        break;
      case 1:
        dtype = DTYPE_CATEGORICAL;
        break;
      default:
        std::cerr
            << "[ERROR] Only Boolean and Categorical variables are supported "
               "now!"
            << std::endl;
        std::abort();
    }
    bool is_evidence = role_serialized >=
                       1;  // TODO interpret as bit vector instead of number?
    bool is_observation = role_serialized == 2;
    variable_value_t init_value = is_evidence ? initial_value : 0;
    variables[vid] = RawVariable(vid, dtype, is_evidence, cardinality,
                                 init_value, init_value, is_observation);
    ++size.num_variables;
    if (is_evidence) {
      ++size.num_variables_evidence;
    } else {
      ++size.num_variables_query;
    }
  }
}

void FactorGraph::load_factors(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  while (file && file.peek() != EOF) {
    uint16_t type;
    factor_arity_t arity;
    // read fields
    read_be_or_die(file, type);
    read_be_or_die(file, arity);
    // register the factor
    factors[size.num_factors] =
        RawFactor(size.num_factors, -1, (factor_function_type_t)type, arity);
    for (factor_arity_t position = 0; position < arity; ++position) {
      // read fields for each variable reference
      variable_id_t variable_id;
      variable_value_t should_equal_to;
      read_be_or_die(file, variable_id);
      read_be_or_die(file, should_equal_to);
      assert(variable_id < capacity.num_variables && variable_id >= 0);
      // add to adjacency lists
      factors[size.num_factors].add_variable_in_factor(
          VariableInFactor(variable_id, position, should_equal_to));
      variables[variable_id].add_factor_id(size.num_factors);
    }
    size.num_edges += arity;
    switch (type) {
      case FUNC_AND_CATEGORICAL: {
        // weight references for categorical factors
        factor_weight_key_t n_weights = 0;
        read_be_or_die(file, n_weights);
        factors[size.num_factors].weight_ids =
            new std::unordered_map<factor_weight_key_t, weight_id_t>(n_weights);
        for (factor_weight_key_t i = 0; i < n_weights; ++i) {
          // calculate radix-based key into weight_ids (see also
          // FactorGraph::get_categorical_weight_id)
          // TODO: refactor the above formula into a shared routine. (See also
          // FactorGraph::get_categorical_weight_id)
          factor_weight_key_t key = 0;
          for (factor_arity_t j = 0; j < arity; ++j) {
            const Variable &var =
                variables[factors[size.num_factors].tmp_variables.at(j).vid];
            variable_value_t value_id;
            read_be_or_die(file, value_id);
            key *= var.cardinality;
            key += var.get_domain_index(value_id);
          }
          weight_id_t wid;
          read_be_or_die(file, wid);
          (*factors[size.num_factors].weight_ids)[key] = wid;
        }
        break;
      }

      default:
        // weight reference is simple for Boolean factors
        weight_id_t wid;
        read_be_or_die(file, wid);
        factors[size.num_factors].weight_id = wid;
    }
    ++size.num_factors;
  }
}

void FactorGraph::load_domains(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  while (file && file.peek() != EOF) {
    // read field to find which categorical variable this block is for
    variable_id_t vid;
    read_be_or_die(file, vid);
    RawVariable &variable = variables[vid];
    // read all category values for this variables
    num_variable_values_t domain_size;
    read_be_or_die(file, domain_size);
    assert(variable.cardinality == domain_size);
    std::vector<variable_value_t> domain_list(domain_size);
    variable.domain_map.reset(
        new std::unordered_map<variable_value_t, variable_value_index_t>());
    for (variable_value_index_t i = 0; i < domain_size; ++i) {
      variable_value_t value;
      read_be_or_die(file, value);
      domain_list[i] = value;
    }
    // populate the mapping from value to index
    // TODO this indexing may be unnecessary (we can use the variable_value_t
    // directly for computing keys
    std::sort(domain_list.begin(), domain_list.end());
    for (variable_value_index_t i = 0; i < domain_size; ++i) {
      (*variable.domain_map)[domain_list[i]] = i;
    }
    // adjust the initial assignments to a valid one instead of zero for query
    // variables
    if (!variable.is_evid) {
      variable.assignment_free = variable.assignment_evid = domain_list[0];
    }
  }
}

/* Also used in test, but not exported in .h */
const std::string get_copy_filename(const std::string &filename, int i) {
  return filename + ".part" + std::to_string(i);
}

void resume(std::string filename, int i, CompiledFactorGraph &cfg) {
  std::ifstream inf;
  inf.open(get_copy_filename(filename, i), std::ios::in | std::ios::binary);

  /* Read metadata */
  inf.read((char *)&cfg.n_var, sizeof(long));
  inf.read((char *)&cfg.n_factor, sizeof(long));
  inf.read((char *)&cfg.n_weight, sizeof(long));
  inf.read((char *)&cfg.n_edge, sizeof(long));

  inf.read((char *)&cfg.c_nvar, sizeof(long));
  inf.read((char *)&cfg.c_nfactor, sizeof(long));
  inf.read((char *)&cfg.c_nweight, sizeof(long));
  inf.read((char *)&cfg.c_edge, sizeof(long));

  inf.read((char *)&cfg.n_evid, sizeof(long));
  inf.read((char *)&cfg.n_query, sizeof(long));

  inf.read((char *)&cfg.stepsize, sizeof(double));

  inf.read((char *)&cfg.is_inc, sizeof(int));

  /*
   * Note that at this point, we have recovered cfg.n_var. Plus, assume
   * that the CompiledFactorGraph has been partially initialized through
   * the graph.meta file, which should at least give us non-null arrays.
   */
  assert(cfg.variables);
  for (auto j = 0; j < cfg.n_var; j++) {
    inf.read((char *)&cfg.variables[j].id, sizeof(long));
    inf.read((char *)&cfg.variables[j].domain_type, sizeof(int));
    inf.read((char *)&cfg.variables[j].is_evid, sizeof(int));
    inf.read((char *)&cfg.variables[j].is_observation, sizeof(int));
    inf.read((char *)&cfg.variables[j].cardinality, sizeof(int));

    inf.read((char *)&cfg.variables[j].assignment_evid, sizeof(VariableValue));
    inf.read((char *)&cfg.variables[j].assignment_free, sizeof(VariableValue));

    inf.read((char *)&cfg.variables[j].n_factors, sizeof(int));
    inf.read((char *)&cfg.variables[j].n_start_i_factors, sizeof(long));

    inf.read((char *)&cfg.variables[j].n_start_i_tally, sizeof(long));

    /* XXX: Ignore last 3 components of Variable, might dump them anyways. */
    /* XXX: What to do about domain_map, though? */
  }

  assert(cfg.factors);
  for (auto j = 0; j < cfg.n_factor; j++) {
    inf.read((char *)&cfg.factors[j].id, sizeof(FactorIndex));
    inf.read((char *)&cfg.factors[j].weight_id, sizeof(WeightIndex));
    inf.read((char *)&cfg.factors[j].func_id, sizeof(int));
    inf.read((char *)&cfg.factors[j].n_variables, sizeof(int));
    inf.read((char *)&cfg.factors[j].n_start_i_vif, sizeof(long));

    /* XXX: Also ignoring weight_ids in Factors */
  }

  assert(cfg.compact_factors);
  assert(cfg.compact_factors_weightids);
  assert(cfg.factor_ids);
  assert(cfg.vifs);
  for (auto j = 0; j < cfg.n_edge; j++) {
    inf.read((char *)&cfg.compact_factors[j].id, sizeof(FactorIndex));
    inf.read((char *)&cfg.compact_factors[j].func_id, sizeof(int));
    inf.read((char *)&cfg.compact_factors[j].n_variables, sizeof(int));
    inf.read((char *)&cfg.compact_factors[j].n_start_i_vif, sizeof(long));

    inf.read((char *)&cfg.compact_factors_weightids[j], sizeof(int));

    inf.read((char *)&cfg.factor_ids[j], sizeof(long));

    inf.read((char *)&cfg.vifs[j].vid, sizeof(long));
    inf.read((char *)&cfg.vifs[j].n_position, sizeof(int));
    inf.read((char *)&cfg.vifs[j].is_positive, sizeof(int));
    inf.read((char *)&cfg.vifs[j].equal_to, sizeof(VariableValue));
  }

  assert(cfg.infrs);
  inf.read((char *)&cfg.infrs->ntallies, sizeof(long));
  for (auto j = 0; j < cfg.infrs->ntallies; j++) {
    inf.read((char *)&cfg.infrs->multinomial_tallies[j], sizeof(long));
  }

  for (auto j = 0; j < cfg.n_var; j++) {
    inf.read((char *)&cfg.infrs->agg_means[j], sizeof(double));
    inf.read((char *)&cfg.infrs->agg_nsamples[j], sizeof(double));
    inf.read((char *)&cfg.infrs->assignments_free[j], sizeof(VariableValue));
    inf.read((char *)&cfg.infrs->assignments_evid[j], sizeof(VariableValue));
  }

  /*
   * XXX: We don't write the weights in `infrs` on purpose, since they're
   * going to be recovered by a separate workflow.
   */

  return;
}

/**
 * For now, assume we use the checkpointing model where we write each factor
 * graph into a separate file.
 *
 * TODO: It's worth explaining this process, and some design decisions that
 * were made in much more detail. I will do that when I clean up the code.
 */
void checkpoint(std::string filename, std::vector<CompiledFactorGraph> &cfgs) {
  auto n_cfgs = cfgs.size();
  for (auto i = 0; i < n_cfgs; i++) {
    std::ofstream outf;
    outf.open(get_copy_filename(filename, i), std::ios::out | std::ios::binary);

    const auto &cfg = cfgs[i];

    /* Now write all the common things that a CompiledFactorGraph has. */
    /* TODO: Need to convert to network order!? */
    outf.write((char *)&cfg.n_var, sizeof(long));
    outf.write((char *)&cfg.n_factor, sizeof(long));
    outf.write((char *)&cfg.n_weight, sizeof(long));
    outf.write((char *)&cfg.n_edge, sizeof(long));

    outf.write((char *)&cfg.c_nvar, sizeof(long));
    outf.write((char *)&cfg.c_nfactor, sizeof(long));
    outf.write((char *)&cfg.c_nweight, sizeof(long));
    outf.write((char *)&cfg.c_edge, sizeof(long));

    outf.write((char *)&cfg.n_evid, sizeof(long));
    outf.write((char *)&cfg.n_query, sizeof(long));

    outf.write((char *)&cfg.stepsize, sizeof(double));

    outf.write((char *)&cfg.is_inc, sizeof(int));

    for (auto j = 0; j < cfg.n_var; j++) {
      outf.write((char *)&cfg.variables[j].id, sizeof(long));
      outf.write((char *)&cfg.variables[j].domain_type, sizeof(int));
      outf.write((char *)&cfg.variables[j].is_evid, sizeof(int));
      outf.write((char *)&cfg.variables[j].is_observation, sizeof(int));
      outf.write((char *)&cfg.variables[j].cardinality, sizeof(int));

      outf.write((char *)&cfg.variables[j].assignment_evid,
                 sizeof(VariableValue));
      outf.write((char *)&cfg.variables[j].assignment_free,
                 sizeof(VariableValue));

      outf.write((char *)&cfg.variables[j].n_factors, sizeof(int));
      outf.write((char *)&cfg.variables[j].n_start_i_factors, sizeof(long));

      outf.write((char *)&cfg.variables[j].n_start_i_tally, sizeof(long));

      /* XXX: Ignore last 3 components of Variable, might dump them anyways. */
      /* XXX: What to do about domain_map, though? */
    }

    for (auto j = 0; j < cfg.n_factor; j++) {
      outf.write((char *)&cfg.factors[j].id, sizeof(FactorIndex));
      outf.write((char *)&cfg.factors[j].weight_id, sizeof(WeightIndex));
      outf.write((char *)&cfg.factors[j].func_id, sizeof(int));
      outf.write((char *)&cfg.factors[j].n_variables, sizeof(int));
      outf.write((char *)&cfg.factors[j].n_start_i_vif, sizeof(long));

      /* XXX: Also ignoring weight_ids in Factors */
    }

    for (auto j = 0; j < cfg.n_edge; j++) {
      outf.write((char *)&cfg.compact_factors[j].id, sizeof(FactorIndex));
      outf.write((char *)&cfg.compact_factors[j].func_id, sizeof(int));
      outf.write((char *)&cfg.compact_factors[j].n_variables, sizeof(int));
      outf.write((char *)&cfg.compact_factors[j].n_start_i_vif, sizeof(long));

      outf.write((char *)&cfg.compact_factors_weightids[j], sizeof(int));

      outf.write((char *)&cfg.factor_ids[j], sizeof(long));

      outf.write((char *)&cfg.vifs[j].vid, sizeof(long));
      outf.write((char *)&cfg.vifs[j].n_position, sizeof(int));
      outf.write((char *)&cfg.vifs[j].is_positive, sizeof(int));
      outf.write((char *)&cfg.vifs[j].equal_to, sizeof(VariableValue));
    }

    outf.write((char *)&cfg.infrs->ntallies, sizeof(long));
    for (auto j = 0; j < cfg.infrs->ntallies; j++) {
      outf.write((char *)&cfg.infrs->multinomial_tallies[j], sizeof(long));
    }

    for (auto j = 0; j < cfg.n_var; j++) {
      outf.write((char *)&cfg.infrs->agg_means[j], sizeof(double));
      outf.write((char *)&cfg.infrs->agg_nsamples[j], sizeof(double));
      outf.write((char *)&cfg.infrs->assignments_free[j],
                 sizeof(VariableValue));
      outf.write((char *)&cfg.infrs->assignments_evid[j],
                 sizeof(VariableValue));
    }

    /*
     * XXX: We don't write the weights in `infrs` on purpose, since they're
     * going to be recovered by a separate workflow.
     */

    outf.close();
  }
}

}  // namespace dd
