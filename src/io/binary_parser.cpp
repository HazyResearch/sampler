#include <iostream>
#include <fstream>
#include <stdint.h>
#include "binary_parser.h"


// 64-bit big endian to little endian
# define bswap_64(x) \
     ((((x) & 0xff00000000000000ull) >> 56)                                   \
      | (((x) & 0x00ff000000000000ull) >> 40)                                 \
      | (((x) & 0x0000ff0000000000ull) >> 24)                                 \
      | (((x) & 0x000000ff00000000ull) >> 8)                                  \
      | (((x) & 0x00000000ff000000ull) << 8)                                  \
      | (((x) & 0x0000000000ff0000ull) << 24)                                 \
      | (((x) & 0x000000000000ff00ull) << 40)                                 \
      | (((x) & 0x00000000000000ffull) << 56))

// 16-bit big endian to little endian
#define bswap_16(x) \
     ((unsigned short int) ((((x) >> 8) & 0xff) | (((x) & 0xff) << 8)))

// Read meta data file, return Meta struct 
Meta read_meta(string meta_file)
{
	ifstream file;
	file.open(meta_file.c_str());
	string buf;
	Meta meta;
	getline(file, buf, ',');
	meta.num_weights = atoll(buf.c_str());
	getline(file, buf, ',');
	meta.num_variables = atoll(buf.c_str());
	getline(file, buf, ',');
	meta.num_factors = atoll(buf.c_str());
	getline(file, buf, ',');
	meta.num_edges = atoll(buf.c_str());
	getline(file, meta.weights_file, ',');
	getline(file, meta.variables_file, ',');
	getline(file, meta.factors_file, ',');
	getline(file, meta.edges_file, ',');
	file.close();
	return meta;
}

// Read weights and load into factor graph
long long read_weights(string filename, dd::FactorGraph &fg)
{
	ifstream file;
    file.open(filename.c_str(), ios::in | ios::binary);
    long long count = 0;
    long long id;
    bool isfixed;
    char padding;
    double initial_value;
    while (file.good()) {
    	// read fields
        file.read((char *)&id, 8);
        file.read((char *)&padding, 1);
        if (!file.read((char *)&initial_value, 8)) break;
        // convert endian
        id = bswap_64(id);
        isfixed = padding;
        long long tmp = bswap_64(*(uint64_t *)&initial_value);
        initial_value = *(double *)&tmp;

        // load into factor graph
        fg.weights[fg.c_nweight] = dd::Weight(id, initial_value, isfixed);
		fg.c_nweight++;
		count++;
    }
    file.close();
    return count;
}


// Read variables
long long read_variables(string filename, dd::FactorGraph &fg)
{
    ifstream file;
    file.open(filename.c_str(), ios::in | ios::binary);
    long long count = 0;
    long long id;
    char isevidence;
    double initial_value;
    short type;
    long long edge_count;
    long long cardinality;
    while (file.good()) {
        // read fields
        file.read((char *)&id, 8);
        file.read((char *)&isevidence, 1);
        file.read((char *)&initial_value, 8);
        file.read((char *)&type, 2);
        file.read((char *)&edge_count, 8);
        if (!file.read((char *)&cardinality, 8)) break;

        // convert endian
        id = bswap_64(id);
        type = bswap_16(type);
        long long tmp = bswap_64(*(uint64_t *)&initial_value);
        initial_value = *(double *)&tmp;
        edge_count = bswap_64(edge_count);
        cardinality = bswap_64(cardinality);

        // printf("----- id=%lli isevidence=%d initial=%f type=%d edge_count=%lli cardinality=%lli\n", id, isevidence, initial_value, type, edge_count, cardinality);

        count++;

        int type_const, upper_bound;
        if (type == 0) {
            type_const  = DTYPE_BOOLEAN;
            upper_bound = 1;
        } else if (type == 1) {
            type_const  = DTYPE_MULTINOMIAL;
            upper_bound = cardinality - 1;
        } else if (type == 2) {
            type_const  = DTYPE_REAL;
            upper_bound = 0;
        } else if (type == 3) {
            type_const = DTYPE_CENSORED_MULTINOMIAL;
            upper_bound = cardinality - 1;
        } else {
            cerr << "[ERROR] Only Boolean and Multinomial variables are supported now!" << endl;
            exit(1);
        }
        bool is_evidence    = isevidence >= 1;
        bool is_observation = isevidence == 2;
        bool is_censored    = isevidence == 3;
        double init_value   = is_evidence ? initial_value : 0;

        fg.variables[fg.c_nvar] = dd::Variable(id, type_const, is_evidence, 0, upper_bound,
            init_value, init_value, edge_count, is_observation, is_censored);
        fg.c_nvar++;
        if (is_evidence) {
            fg.n_evid++;
        } else {
            fg.n_query++;
        }
    }
    file.close();

    return count;
}

long long read_factors(string filename, dd::FactorGraph &fg)
{
    ifstream file;
    file.open(filename.c_str(), ios::in | ios::binary);
    long long count = 0;
    long long id;
    long long weightid;
    short type;
    long long edge_count;
    while (file.good()) {
        file.read((char *)&id, 8);
        file.read((char *)&weightid, 8);
        file.read((char *)&type, 2);
        if (!file.read((char *)&edge_count, 8)) break;

        id = bswap_64(id);
        weightid = bswap_64(weightid);
        type = bswap_16(type);
        edge_count = bswap_64(edge_count);

        count++;
        fg.factors[fg.c_nfactor] = dd::Factor(id, weightid, type, edge_count);
        fg.c_nfactor ++;
    }
    file.close();
    return count;
}

long long read_edges(string filename, dd::FactorGraph &fg)
{
    ifstream file;
    file.open(filename.c_str(), ios::in | ios::binary);
    long long count = 0;
    long long variable_id;
    long long factor_id;
    long long position;
    bool ispositive;
    char padding;
    long long equal_predicate;
    while (file.good()) {
        // read fields
        file.read((char *)&variable_id, 8);
        file.read((char *)&factor_id, 8);
        file.read((char *)&position, 8);
        file.read((char *)&padding, 1);
        if (!file.read((char *)&equal_predicate, 8)) break;
        
        variable_id = bswap_64(variable_id);
        factor_id = bswap_64(factor_id);
        position = bswap_64(position);
        ispositive = padding;
        equal_predicate = bswap_64(equal_predicate);
        
        count++;
        // printf("varid=%lli, factorid=%lli, position=%lli, predicate=%lli\n", variable_id, factor_id, position, equal_predicate);

        //std::cout << variable_id << std::endl;

        // wrong id
    	if(variable_id >= fg.n_var || variable_id < 0){
    	  assert(false);
    	}

    	if(factor_id >= fg.n_factor || factor_id < 0){
    	  std::cout << "wrong fid = " << factor_id << std::endl;
    	  assert(false);
        }

        // add variables to factors
        if (fg.variables[variable_id].domain_type == DTYPE_BOOLEAN) {
            fg.factors[factor_id].tmp_variables.push_back(
                dd::VariableInFactor(variable_id, fg.variables[variable_id].upper_bound, variable_id, position, ispositive));
        } else {
            fg.factors[factor_id].tmp_variables.push_back(
                dd::VariableInFactor(variable_id, position, ispositive, equal_predicate));
        }
        fg.variables[variable_id].tmp_factor_ids.push_back(factor_id);

    }
    file.close();
    return count;   
}

void read_cnn_ports(string filename, dd::FactorGraph &fg) {
    ifstream file(filename.c_str());
    std::string port;
    while (file >> port) {
        fg.cnn_ports.push_back(port);
    }
}

void read_cnn_configs(string filename, dd::FactorGraph &fg) {
    ifstream file(filename.c_str());
    long n_evid;
    int cnn_train_iteration;
    int cnn_test_iteration;
    int cnn_test_interval;
    int cnn_batch_size;
    file >> n_evid >> cnn_train_iteration >> cnn_test_iteration >> cnn_test_interval >> cnn_batch_size;
    fg.cnn_n_evid.push_back(n_evid);
    fg.cnn_train_iterations.push_back(cnn_train_iteration);
    fg.cnn_test_iterations.push_back(cnn_test_iteration);
    fg.cnn_test_intervals.push_back(cnn_test_interval);
    fg.cnn_batch_sizes.push_back(cnn_batch_size);
    fg.cnn_is_pretrained.push_back(cnn_batch_size == 0);
    file.close();
}

