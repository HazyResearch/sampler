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
long long read_weights(string filename, dd::FactorGraph &fg, long long n_weight)
{
    long long count = 0;
    long long id;
    bool isfixed;
    char padding;
    double initial_value;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = n_weight * WEIGHT_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&padding, ptr + 8, 1);
        memcpy((char *)&initial_value, ptr + 9, 8);
        ptr += WEIGHT_RECORD_SIZE;

        isfixed = padding;
        // convert endian
        id = bswap_64(id);
        long long tmp = bswap_64(*(uint64_t *)&initial_value);
        initial_value = *(double *)&tmp;
        // load into factor graph
        fg.weights[fg.c_nweight] = dd::Weight(id, initial_value, isfixed);
        fg.c_nweight++;
        count++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

// Read variables
long long read_variables(string filename, dd::FactorGraph &fg, long long n_variable)
{
    long long count = 0;
    long long id;
    char isevidence;
    double initial_value;
    short type;
    long long edge_count;
    long long cardinality;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = n_variable * VARIABLE_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&isevidence, ptr + 8, 1);
        memcpy((char *)&initial_value, ptr + 9, 8);
        memcpy((char *)&type, ptr + 17, 2);
        memcpy((char *)&edge_count, ptr + 19, 8);
        memcpy((char *)&cardinality, ptr + 27, 8);
        ptr += VARIABLE_RECORD_SIZE;
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
        } else {
            cerr << "[ERROR] Only Boolean and Multinomial variables are supported now!" << endl;
            exit(1);
        }
        bool is_evidence    = isevidence >= 1;
        bool is_observation = isevidence == 2;
        double init_value   = is_evidence ? initial_value : 0;

        fg.variables[fg.c_nvar] = dd::Variable(id, type_const, is_evidence, 0, upper_bound,
            init_value, init_value, edge_count, is_observation);
        fg.c_nvar++;
        if (is_evidence) {
            fg.n_evid++;
        } else {
            fg.n_query++;
        }
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

// Read factors (original mode)
// The format of each line in factor file is: weight_id, type, equal_predicate, edge_count, variable_id_1, padding_1, ..., variable_id_k, padding_k
// It is binary format without delimiter.
long long read_factors(string filename, dd::FactorGraph &fg, long long n_factor, long long n_edge)
{
    long long count = 0;
    long long variable_id;
    long long weightid;
    short type;
    long long edge_count;
    long long equal_predicate;
    char padding;
    bool ispositive;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = n_factor * FACTOR_RECORD_SIZE + n_edge * EDGE_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&weightid, ptr, 8);
        memcpy((char *)&type, ptr + 8, 2);
        memcpy((char *)&equal_predicate, ptr + 10, 8);
        memcpy((char *)&edge_count, ptr + 18, 8);
        ptr += FACTOR_RECORD_SIZE;

        weightid = bswap_64(weightid);
        type = bswap_16(type);
        edge_count = bswap_64(edge_count);
        equal_predicate = bswap_64(equal_predicate);

        count++;
        fg.factors[fg.c_nfactor] = dd::Factor(fg.c_nfactor, weightid, type, edge_count);

        for (long long position = 0; position < edge_count; position++) {
            memcpy((char *)&variable_id, ptr, 8);
            memcpy((char *)&padding, ptr + 8, 1);
            ptr += EDGE_RECORD_SIZE;
            variable_id = bswap_64(variable_id);
            ispositive = padding;

            // wrong id
            if(variable_id >= fg.n_var || variable_id < 0){
                assert(false);
            }

            fg.edges[fg.c_nedge] = dd::Edge(variable_id, fg.c_nfactor, position, ispositive, equal_predicate);
            fg.c_nedge++;
        }
        fg.c_nfactor++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

// Read factors (incremental mode)
// The format of each line in factor file is: factor_id, weight_id, type, edge_count
// It is binary format without delimiter.
long long read_factors_inc(string filename, dd::FactorGraph &fg, long long n_factor)
{
    long long count = 0;
    long long id;
    long long weightid;
    short type;
    long long edge_count;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = n_factor * FACTOR_INC_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        memcpy((char *)&id, ptr, 8);
        memcpy((char *)&weightid, ptr + 8, 8);
        memcpy((char *)&type, ptr + 16, 2);
        memcpy((char *)&edge_count, ptr + 18, 8);
        ptr += FACTOR_INC_RECORD_SIZE;

        id = bswap_64(id);
        weightid = bswap_64(weightid);
        type = bswap_16(type);
        edge_count = bswap_64(edge_count);

        count++;
        fg.factors[fg.c_nfactor] = dd::Factor(id, weightid, type, edge_count);
        fg.c_nfactor++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}

// Read edges (incremental mode)
// The format of each line in factor file is: variable_id, factor_id, position, padding
// It is binary format without delimiter.
long long read_edges_inc(string filename, dd::FactorGraph &fg, long long n_edge)
{
    long long count = 0;
    long long variable_id;
    long long factor_id;
    long long position;
    bool ispositive;
    char padding;
    long long equal_predicate;

    int fd = open(filename.c_str(), O_RDONLY);
    long size = n_edge * EDGE_INC_RECORD_SIZE;
    char *memory_map = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char *ptr = memory_map;
    char *end = memory_map + size;
    while (ptr < end) {
        // read fields
        memcpy((char *)&variable_id, ptr, 8);
        memcpy((char *)&factor_id, ptr + 8, 8);
        memcpy((char *)&position, ptr + 16, 8);
        memcpy((char *)&padding, ptr + 24, 1);
        memcpy((char *)&equal_predicate, ptr + 25, 8);
        ptr += EDGE_INC_RECORD_SIZE;

        variable_id = bswap_64(variable_id);
        factor_id = bswap_64(factor_id);
        position = bswap_64(position);
        ispositive = padding;
        equal_predicate = bswap_64(equal_predicate);

        count++;
        // printf("varid=%lli, factorid=%lli, position=%lli, predicate=%lli\n", variable_id, factor_id, position, equal_predicate);

        // wrong id
        if(variable_id >= fg.n_var || variable_id < 0){
          assert(false);
        }

        if(factor_id >= fg.n_factor || factor_id < 0){
          std::cout << "wrong fid = " << factor_id << std::endl;
          assert(false);
        }

        fg.edges[fg.c_nedge] = dd::Edge(variable_id, factor_id, position, ispositive, equal_predicate);
        fg.c_nedge++;
    }
    munmap(memory_map, size);
    close(fd);
    return count;
}
