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
#include <sys/stat.h>

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
// const std::string get_copy_filename(const std::string &filename, int i) {
//  return filename + ".part" + std::to_string(i);
//}

/**
 * For now, assume we use the checkpointing model where we write each factor
 * graph into a separate file.
 *
 * TODO: It's worth explaining this process, and some design decisions that
 * were made in much more detail. I will do that when I clean up the code.
 */
void CompactFactorGraph::dump(const std::string &snapshot_path) {
  constexpr char snapshot_filename[] = "graph.checkpoint";

  // XXX: We officially do NOT support Windows.
  std::ofstream outf(snapshot_path + "/" + snapshot_filename,
            std::ios::out | std::ios::binary);
  if (!outf.is_open()) {
    std::cout << "Failed to create snapshot file" << snapshot_path
              << std::endl;
    std::abort();
  }


  for (auto j = 0; j < size.num_variables; j++) {
    outf.write((char *)&variables[j].id, sizeof(variable_id_t));
    outf.write((char *)&variables[j].domain_type,
               sizeof(variable_domain_type_t));

    outf.write((char *)&variables[j].is_evid, sizeof(int));
    outf.write((char *)&variables[j].is_observation, sizeof(int));
    outf.write((char *)&variables[j].cardinality,
               sizeof(num_variable_values_t));

    outf.write((char *)&variables[j].assignment_evid, sizeof(variable_value_t));
    outf.write((char *)&variables[j].assignment_free, sizeof(variable_value_t));

    outf.write((char *)&variables[j].n_factors, sizeof(num_edges_t));
    outf.write((char *)&variables[j].n_start_i_factors, sizeof(num_edges_t));

    outf.write((char *)&variables[j].n_start_i_tally, sizeof(num_samples_t));

    /* XXX: Ignore last 3 components of Variable, might dump them anyways. */
    /* XXX: What to do about domain_map, though? */
  }

  for (auto j = 0; j < size.num_factors; j++) {
    outf.write((char *)&factors[j].id, sizeof(factor_id_t));
    outf.write((char *)&factors[j].weight_id, sizeof(weight_id_t));
    outf.write((char *)&factors[j].func_id, sizeof(factor_function_type_t));

    outf.write((char *)&factors[j].n_variables, sizeof(factor_arity_t));
    outf.write((char *)&factors[j].n_start_i_vif, sizeof(num_edges_t));

    /* XXX: Also ignoring weight_ids in Factors */
  }

  for (auto j = 0; j < size.num_edges; j++) {
    outf.write((char *)&compact_factors[j].id, sizeof(factor_id_t));
    outf.write((char *)&compact_factors[j].func_id,
               sizeof(factor_function_type_t));

    outf.write((char *)&compact_factors[j].n_variables, sizeof(factor_arity_t));
    outf.write((char *)&compact_factors[j].n_start_i_vif, sizeof(num_edges_t));

    outf.write((char *)&compact_factors_weightids[j], sizeof(weight_id_t));

    outf.write((char *)&factor_ids[j], sizeof(factor_id_t));

    outf.write((char *)&vifs[j].vid, sizeof(variable_id_t));
    outf.write((char *)&vifs[j].n_position, sizeof(factor_arity_t));
    outf.write((char *)&vifs[j].equal_to, sizeof(variable_value_t));
  }

  outf.close();
}

void CompactFactorGraph::resume(const std::string &snapshot_path) {
  constexpr char snapshot_filename[] = "graph.checkpoint";

  std::ifstream inf(snapshot_path + "/" + snapshot_filename,
           std::ios::in | std::ios::binary);
  if (!inf.is_open()) {
    std::cout << "Error while opening snapshot file " << std::endl;
    std::abort();
  }

  /*
   * Note that at this point, we have recovered cfg.n_var. Plus, assume
   * that the CompactFactorGraph has been partially initialized through
   * the graph.meta file, which should at least give us non-null arrays.
   */
  for (auto j = 0; j < size.num_variables; j++) {
    inf.read((char *)&variables[j].id, sizeof(variable_id_t));
    inf.read((char *)&variables[j].domain_type,
             sizeof(variable_domain_type_t));

    inf.read((char *)&variables[j].is_evid, sizeof(int));
    inf.read((char *)&variables[j].is_observation, sizeof(int));
    inf.read((char *)&variables[j].cardinality, sizeof(num_variable_values_t));

    inf.read((char *)&variables[j].assignment_evid, sizeof(variable_value_t));
    inf.read((char *)&variables[j].assignment_free, sizeof(variable_value_t));

    inf.read((char *)&variables[j].n_factors, sizeof(num_edges_t));
    inf.read((char *)&variables[j].n_start_i_factors, sizeof(num_edges_t));

    inf.read((char *)&variables[j].n_start_i_tally, sizeof(num_samples_t));

    /* XXX: Ignore last 3 components of Variable, might dump them anyways. */
    /* XXX: What to do about domain_map, though? */
  }

  for (auto j = 0; j < size.num_factors; j++) {
    inf.read((char *)&factors[j].id, sizeof(factor_id_t));
    inf.read((char *)&factors[j].weight_id, sizeof(weight_id_t));
    inf.read((char *)&factors[j].func_id, sizeof(factor_function_type_t));
    inf.read((char *)&factors[j].n_variables, sizeof(factor_arity_t));

    inf.read((char *)&factors[j].n_start_i_vif, sizeof(num_edges_t));

    /* XXX: Also ignoring weight_ids in Factors */
  }

  for (auto j = 0; j < size.num_edges; j++) {
    inf.read((char *)&compact_factors[j].id, sizeof(factor_id_t));
    inf.read((char *)&compact_factors[j].func_id,
             sizeof(factor_function_type_t));
    inf.read((char *)&compact_factors[j].n_variables, sizeof(factor_arity_t));
    inf.read((char *)&compact_factors[j].n_start_i_vif, sizeof(num_edges_t));

    inf.read((char *)&compact_factors_weightids[j], sizeof(weight_id_t));

    inf.read((char *)&factor_ids[j], sizeof(factor_id_t));

    inf.read((char *)&vifs[j].vid, sizeof(variable_id_t));
    inf.read((char *)&vifs[j].n_position, sizeof(factor_arity_t));
    inf.read((char *)&vifs[j].equal_to, sizeof(variable_value_t));
  }

  /*
   * NOTE: We don't write the weights in `infrs` on purpose, since they're
   * going to be recovered by a separate workflow.
   */

  return;
}

}  // namespace dd
