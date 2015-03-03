#include <iostream>
#include <fstream>
#include <stdint.h>
#include "binary_parser.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <unistd.h>


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

#define CHANGE_BYTE_ORDER 1

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
    std::cout << "LOADING WEIGHTS..." << std::endl;
    long long count = 0;
    long long id;
    bool isfixed;
    char padding;
    double initial_value;
    int fd = open(filename.c_str(), O_RDONLY);
    long size = fg.n_weight * WEIGHT_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&padding, ptr + 8, 1);
        memcpy((char *)&initial_value, ptr + 9, 8);
        ptr += WEIGHT_RECORD_SIZE;
        count++;

        isfixed = padding;
        // convert endian
        id = bswap_64(id);
        long tmp = bswap_64(*(uint64_t *)&initial_value);
        initial_value = *(double *)&tmp;
        // load into factor graph
        fg.weights[fg.c_nweight] = dd::Weight(id, initial_value, isfixed);
        fg.c_nweight++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}


// Read variables
long long read_variables(string filename, dd::FactorGraph &fg)
{
    std::cout << "LOADING VARIABLES..." << std::endl;
    long long count = 0;
    long long id;
    bool isevidence;
    char padding1;
    double initial_value;
    short type;
    long long edge_count;
    long long cardinality;
    int fd = open(filename.c_str(), O_RDONLY);
    long size = fg.n_var * VARIABLE_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&padding1, ptr + 8, 1);
        memcpy((char *)&initial_value, ptr + 9, 8);
        memcpy((char *)&type, ptr + 17, 2);
        memcpy((char *)&edge_count, ptr + 19, 8);
        memcpy((char *)&cardinality, ptr + 27, 8);
        ptr += VARIABLE_RECORD_SIZE;
        count++;

        isevidence = padding1;
        id = bswap_64(id);
        type = bswap_16(type);
        long long tmp = bswap_64(*(uint64_t *)&initial_value);
        initial_value = *(double *)&tmp; 
        edge_count = bswap_64(edge_count);
        cardinality = bswap_64(cardinality);
        // printf("----- id=%lli isevidence=%d initial=%f type=%d edge_count=%lli cardinality=%lli\n", id, isevidence, initial_value, type, edge_count, cardinality);

        // add to factor graph
        if (type == 0){ // boolean
            if (isevidence) {
                fg.variables[fg.c_nvar] = dd::Variable(id, DTYPE_BOOLEAN, true, 0, 1, 
                    initial_value, initial_value, edge_count);
                fg.c_nvar++;
                fg.n_evid++;
            } else {
                fg.variables[fg.c_nvar] = dd::Variable(id, DTYPE_BOOLEAN, false, 0, 1, 
                    0, 0, edge_count);
                fg.c_nvar++;
                fg.n_query++;
            }
        } else if (type == 1) { // multinomial
            if (isevidence) {
                fg.variables[fg.c_nvar] = dd::Variable(id, DTYPE_MULTINOMIAL, true, 0, 
                    cardinality-1, initial_value, initial_value, edge_count);
                fg.c_nvar ++;
                fg.n_evid ++;
            } else {
                fg.variables[fg.c_nvar] = dd::Variable(id, DTYPE_MULTINOMIAL, false, 0, 
                    cardinality-1, 0, 0, edge_count);
                fg.c_nvar ++;
                fg.n_query ++;
            }
        }else {
            cout << "[ERROR] Only Boolean and Multinomial variables are supported now!" << endl;
            exit(1);
        }
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

long long read_factors(string filename, dd::FactorGraph &fg)
{
    std::cout << "LOADING FACTORS" << std::endl;
    long long count = 0;
    long long id;
    long long weightid;
    short type;
    long long edge_count;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = fg.n_factor * FACTOR_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&weightid, ptr + 8, 8);
        memcpy((char *)&type, ptr + 16, 2);
        memcpy((char *)&edge_count, ptr + 18, 8);
        ptr += FACTOR_RECORD_SIZE;
        count++;

        id = bswap_64(id);
        weightid = bswap_64(weightid);
        type = bswap_16(type);
        edge_count = bswap_64(edge_count);

        fg.factors[fg.c_nfactor] = dd::Factor(id, weightid, type, edge_count);
        fg.c_nfactor++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

long long read_edges(string filename, dd::FactorGraph &fg)
{
    std::cout << "LOADING EDGES..." << std::endl;
    ifstream file;
    file.open(filename.c_str(), ios::in | ios::binary);
    long long count = 0;
    long long variable_id;
    long long factor_id;
    long long position;
    bool ispositive;
    char padding;
    long long equal_predicate;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = fg.n_edge * EDGE_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&variable_id, ptr, 8);
        memcpy((char *)&factor_id, ptr + 8, 8);
        memcpy((char *)&position, ptr + 16, 8);
        memcpy((char *)&padding, ptr + 24, 1);
        memcpy((char *)&equal_predicate, ptr + 25, 8);
        ptr += EDGE_RECORD_SIZE;
        count++;

        variable_id = bswap_64(variable_id);
        factor_id = bswap_64(factor_id);
        position = bswap_64(position);
        ispositive = padding;
        equal_predicate = bswap_64(equal_predicate);

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
    munmap(memory_map, size);
    close(fd);
    return count;  
}

