#!/usr/bin/env python

#### IMPORTS ####
from __future__ import print_function
import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import struct
import math
import random
import sys
import argparse
import mmap
import threading

#### HELPER METHODS ####
def dataType(i):
  return {0: "Boolean",
          1: "Categorical"}.get(i, "Unknown")

#### DEFINE GLOBAL DATA TYPES ####
Meta = np.dtype([('weights',        np.int64),
                 ('variables',      np.int64),
                 ('factors',        np.int64),
                 ('edges',          np.int64)])

Weight  = np.dtype([("isFixed",      np.bool),
                    ("initialValue", np.float64)])

Variable  = np.dtype([("isEvidence",   np.int8),
                      ("initialValue", np.int32),
                      ("dataType",     np.int16),
                      ("cardinality",  np.int32)])

Factor  = np.dtype([("factorFunction", np.int16),
                    ("weightId",       np.int64),
                    ("featureValue",   np.float64), # TODO: This is not used yet
                   ])

FactorStart    = np.dtype(np.int64)
FactorMap      = np.dtype(np.int64)
EqualPredicate = np.dtype(np.int32)
VariableStart  = np.dtype(np.int64)
VariableMap    = np.dtype(np.int64)

#### INIT GLOBAL DATA STRUCTURES ####
# These structures hold the factor graph information
_meta_      = np.array((1,1,1,1),dtype=Meta)
_meta_	    = _meta_[()]

_weights_   = np.empty(0,Weight)
_variables_ = np.empty(0,Variable)
_factors_   = np.empty(0,Factor)
_fstart_    = np.empty(0,FactorStart)
_fmap_      = np.empty(0,FactorMap)
_vstart_    = np.empty(0,VariableStart)
_vmap_      = np.empty(0,VariableMap)
_equalPred_ = np.empty(0,EqualPredicate)

# These structures are needed for the sampling
_Z_	       = np.zeros(0)
_count_	       = np.zeros(0,dtype=np.int64)
_varCopies_    = np.zeros(0,np.int32)
_weightCopies_ = np.zeros(0,np.float64)

#### DEFINE NUMBA TYPES ####
Weight_ = numba.from_dtype(Weight)
Variable_ = numba.from_dtype(Variable)
Factor_ = numba.from_dtype(Factor)

#### DEFINE SAMPLER METHODS ####

### INITIALIZATION METHOD ###
def samplerInit(var_copies=1, weight_copies=1, def_param =(_weights_, _variables_, _Z_, _count_, _varCopies_, _weightCopies_)):

    _count_.resize(_variables_.shape[0], refcheck=False)
  
    _varCopies_.resize((var_copies, _variables_.shape[0]), refcheck=False)
    for i in range(var_copies):
            for j in range(_variables_.shape[0]):
                _varCopies_[i][j] = _variables_[j]["initialValue"]

    _weightCopies_.resize((weight_copies, _weights_.shape[0]), refcheck=False)
    for i in range(weight_copies):
            for j in range(_weights_.shape[0]):
                _weightCopies_[i][j] = _weights_[j]["initialValue"]

    cardinality = 0
    for v in _variables_:
    	cardinality = max(cardinality, v["cardinality"])
    _Z_.resize(cardinality, refcheck=False)

### INFERENCE METHODS ###
def gibbs(nthreads, sweeps, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
	assert(nthreads > 0)
	activeThreads = []
	for threadID in range(nthreads):
		t = threading.Thread(target=gibbsthread, args=[threadID, nthreads, sweeps, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_])
		activeThreads.append(t)
		t.start()
	for i in activeThreads:
		t.join()

	for i in range(0, _variables_.shape[0], max(1, _variables_.shape[0] / 100)):
            print("Var", i + 1, "/", len(_variables_), ":", _count_[i])
        print()

        bins = 10
        hist = np.zeros(bins, dtype=np.int64)
        for i in range(len(_count_)):
            hist[min(_count_[i] * bins / sweeps, bins - 1)] += 1
        for i in range(bins):
            print(i, hist[i])
	


@jit(nopython=True,cache=True,nogil=True)
def gibbsthread(shardID, nshards, sweeps, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        # Indentify start and end variable

	nvar  = _variables_.shape[0]
	start = ((nvar / nshards) + 1) * shardID
	end   = min(((nvar / nshards) + 1) * (shardID + 1), nvar)
        
        # TODO: give option do not store result, or just store tally
        for sweep in range(sweeps):
            for var_samp in range(start,end):
                _count_[var_samp] += sample(var_samp, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)


@jit(nopython=True,cache=True,nogil=True)
def sample(var_samp, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        # TODO: return if is observation
        # TODO: return if is evidence and not sampling evidence
        if _variables_[var_samp]["isEvidence"] != 0:
            return _varCopies_[var_copy][var_samp]

        _varCopies_[var_copy][var_samp] = draw_sample(var_samp, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
        return _varCopies_[var_copy][var_samp]


@jit(nopython=True,cache=True,nogil=True)
def draw_sample(var_samp, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        cardinality = _variables_[var_samp]["cardinality"]
        for value in range(cardinality):
            _Z_[value] = math.exp(potential(var_samp, value, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_))

        for j in range(1, cardinality):
            _Z_[j] += _Z_[j - 1]

        z = random.random() * _Z_[cardinality - 1]
        # TODO: I think this looks at the full vector, will be slow if one var has high cardinality
        return np.argmax(_Z_ >= z)

@jit(nopython=True,cache=True,nogil=True)
def potential(var_samp, value, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        p = 0.0
        for k in range(_vstart_[var_samp], _vstart_[var_samp + 1]):
            factor_id = _vmap_[k]
            p += _weightCopies_[weight_copy][_factors_[_vmap_[k]]["weightId"]]*eval_factor(factor_id, var_samp, value, var_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
        return p

@jit(nopython=True,cache=True,nogil=True)
def eval_factor(factor_id, var_samp, value, var_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        if _factors_[factor_id]["factorFunction"] == 3: # FUNC_EQUAL
            v = value if (_fmap_[_fstart_[factor_id]] == var_samp) else _varCopies_[var_copy][_fmap_[_fstart_[factor_id]]]
            for l in range(_fstart_[factor_id] + 1, _fstart_[factor_id + 1]):
                w = value if (_fmap_[l] == var_samp) else _varCopies_[var_copy][_fmap_[l]]
                if v != w:
                    return -1
            return 1
        elif _factors_[factor_id]["factorFunction"] == 4: # FUNC_ISTRUE
            for l in range(_fstart_[factor_id], _fstart_[factor_id + 1]):
                v = value if (_fmap_[l] == var_samp) else _varCopies_[var_copy][_fmap_[l]]
                if v == 0:
                    return -1
            return 1
        else: # FUNC_UNDEFINED
            print("Error: Factor Function", _factors_[factor_id]["factorFunction"], "( used in factor", factor_id, ") is not implemented.")
            raise NotImplementedError("Factor function is not implemented.")


### LEARNING METHODS ###
@jit(nopython=True,cache=True,nogil=True)
def learn(sweeps, step, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        for sweep in range(sweeps):
            for var_samp in range(_variables_.shape[0]):
                sample_and_sgd(var_samp, step, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
            print(sweep + 1)
            print("Weights:")
            for (i, w) in enumerate(_weights_):
                print("    weightId:", i)
                print("        isFixed:", w["isFixed"])
                print("        weight: ", _weightCopies_[weight_copy][i])
            print()

@jit(nopython=True,cache=True,nogil=True)
def sample_and_sgd(var_samp, step, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_):
        # TODO: return none or sampled var?

        # TODO: return if is observation
        if (_variables_[var_samp]["isEvidence"] == 2):
            return

        _varCopies_[var_copy][var_samp] = draw_sample(var_samp, var_copy, weight_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)

        # TODO: set initialValue
        # TODO: if isevidence or learn_non_evidence
        if _variables_[var_samp]["isEvidence"] == 1:
            for i in range(_vstart_[var_samp], _vstart_[var_samp + 1]):
                factor_id = _vmap_[i]
                weight_id = _factors_[factor_id]["weightId"]

                if not _weights_[weight_id]["isFixed"]:
                    # TODO: save time by checking if initialValue and value are equal first?
                    p0 = eval_factor(factor_id, var_samp, _variables_[var_samp]["initialValue"], var_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
                    p1 = eval_factor(factor_id, var_samp, _varCopies_[var_copy][var_samp], var_copy, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
                    _weightCopies_[weight_copy][weight_id] += step * (p0 - p1)

#### DEFINE NUMBA-BASED DATA LOADING HELPER METHODS ####
@jit(nopython=True,cache=True)
def compute_var_map(fstart, fmap, vstart, vmap):
    for i in fmap:
        vstart[i + 1] += 1

    for i in range(len(vstart) - 1):
        vstart[i + 1] += vstart[i]
    index = vstart.copy()

    for i in range(len(fstart) - 1):
        for j in range(fstart[i], fstart[i + 1]):
            vmap[index[fmap[j]]] = i
            index[fmap[j]] += 1

@jit(nopython=True,cache=True)
def reverse(data, start, end):
    end -= 1
    while (start < end):
        data[start], data[end] = data[end], data[start]
        start += 1
        end -= 1

#### DEFINE NUMBA-BASED DATA LOADING METHODS #### 
@jit(nopython=True,cache=True)
def load_weights(data, nweights, weights):
    for i in range(nweights):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        reverse(data, 17 * i, 17 * i + 8)
        weightId = np.frombuffer(data[(17 * i):(17 * i + 8)], dtype=np.int64)[0]
        #weightId = np.frombuffer(data[(17 * i):(17 * i + 8):-1], dtype=np.int64)[0]
        isFixed      =               data[17 * i + 8]
        reverse(data, 17 * i + 9, 17 * i + 17)
        initialValue = np.frombuffer(data[(17 * i + 9):(17 * i + 17)], dtype=np.float64)[0]

        weights[weightId]["isFixed"] = isFixed
        weights[weightId]["initialValue"] = initialValue
    print("DONE WITH WEIGHTS")

@jit(nopython=True,cache=True)
def load_variables(data, nvariables, variables):
    for i in range(nvariables):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian


        reverse(data, 19 * i, 19 * i + 8)
        variableId   = np.frombuffer(data[(19 * i):(19 * i + 8)], dtype=np.int64)[0]
        #variableId = variableId.byteswap()
        isEvidence   =               data[19 * i + 8]
        reverse(data, 19 * i + 9, 19 * i + 13)
        initialValue = np.frombuffer(data[(19 * i + 9):(19 * i + 13)], dtype=np.int32)[0]
        reverse(data, 19 * i + 13, 19 * i + 15)
        dataType     = np.frombuffer(data[(19 * i + 13):(19 * i + 15)], dtype=np.int16)[0]
        reverse(data, 19 * i + 15, 19 * i + 19)
        cardinality  = np.frombuffer(data[(19 * i + 15):(19 * i + 19)], dtype=np.int32)[0]

        variables[variableId]["isEvidence"] = isEvidence
        variables[variableId]["initialValue"] = initialValue
        variables[variableId]["dataType"] = dataType
        variables[variableId]["cardinality"] = cardinality
    print("DONE WITH VARS")

@jit(nopython=True,cache=True)
def load_factors(data, nfactors, factors, fstart, fmap, equalPredicate):
    index = 0
    for i in range(nfactors):
        reverse(data, index, index + 2)
        factors[i]["factorFunction"] = np.frombuffer(data[index:(index + 2)], dtype=np.int16)[0]

        reverse(data, index + 2, index + 6)
        arity = np.frombuffer(data[(index + 2):(index + 6)], dtype=np.int32)[0]

        index += 6 # TODO: update index once per loop?

        fstart[i + 1] = fstart[i] + arity
        for j in range(arity):
            reverse(data, index, index + 8)
            fmap[fstart[i] + j] = np.frombuffer(data[index:(index + 8)], dtype=np.int64)[0]
            reverse(data, index + 8, index + 12)
            equalPredicate[fstart[i] + j] = np.frombuffer(data[(index + 8):(index + 12)], dtype=np.int32)[0]
            index += 12

        # TODO: handle FUNC_AND_CATEGORICAL
        reverse(data, index, index + 8)
        factors[i]["weightId"]     = np.frombuffer(data[index:(index + 8)], dtype=np.int64)[0]
        reverse(data, index + 8, index + 16)
        factors[i]["featureValue"] = np.frombuffer(data[(index + 8):(index + 16)], dtype=np.float64)[0]
        index += 16
        #variableId1     long    8
        #isPositive1     bool    1
        #variableId2     long    8
        #isPositive2     bool    1
        #info += [(factor_type, arity, variable_id.tolist(), equal_predicate.tolist(), weight_id, feature_value)]
    print("DONE WITH FACTORS")

#### DEFINE PYTHON-BASED DATA LOADING ####
def load(directory=".",
         metafile="graph.meta",
         weightfile="graph.weights",
         variablefile="graph.variables",
         factorfile="graph.factors",
         print_info=False,
         print_only_meta=False,
	 def_param =(_meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_)):

    # TODO: check that entire file is read (nothing more/nothing less)
    # TODO: make error when file does not exist less dumb

    ## LOAD FACTOR GRAPH METADATA ##
    _tmpmeta_ = np.loadtxt(directory + "/" + metafile,
                      delimiter=',',
                      dtype=Meta)
    _tmpmeta_ = _tmpmeta_[()] # convert from 0-dimensional ndarray to scalar
    _meta_['weights'] = _tmpmeta_['weights']
    _meta_['variables'] = _tmpmeta_['variables']
    _meta_['factors'] = _tmpmeta_['factors']
    _meta_['edges'] = _tmpmeta_['edges']
    if print_info:
        print("Meta:")
        print("    weights:  ", _meta_["weights"])
        print("    variables:", _meta_["variables"])
        print("    factors:  ", _meta_["factors"])
        print("    edges:    ", _meta_["edges"])
        print()

    ## LOAD FACTOR GRAPH WEIGHTS ##
    weight_data = np.memmap(directory + "/" + weightfile, mode="c")
    _weights_.resize(_meta_["weights"], refcheck=False)
    # Call numba-based weight loader
    load_weights(weight_data, _meta_["weights"], _weights_)
    #weight.byteswap() # TODO: only if system is little-endian
    if print_info and not print_only_meta:
        print("Weights:")
        for (i, w) in enumerate(_weights_):
            print("    weightId:", i)
            print("        isFixed:", w["isFixed"])
            print("        weight: ", w["initialValue"])
        print()

    ## LOAD FACTOR GRAPH VARIABLES ##
    variable_data = np.memmap(directory + "/" + variablefile, mode="c")
    _variables_.resize(_meta_["variables"], refcheck=False)
    load_variables(variable_data, _meta_["variables"], _variables_)
    #variable.byteswap() # TODO: only if system is little-endian
    if print_info and not print_only_meta:
        print("Variables:")
        for (i, v) in enumerate(_variables_):
            print("    variableId:", i)
            print("        isEvidence:  ", v["isEvidence"])
            print("        initialValue:", v["initialValue"])
            print("        dataType:    ", v["dataType"], "(", dataType(v["dataType"]), ")")
            print("        cardinality: ", v["cardinality"])
            # TODO: print connected factors and num factors
            print()

    ## LOAD FACTOR GRAPH FACTORS ##
    # TODO: might need to sort by factorId? (or just load in right spot)
    factor_data = np.memmap(directory + "/" + factorfile, mode="c")
    _factors_.resize(_meta_["factors"], refcheck=False)
    _fstart_.resize(_meta_["factors"] + 1, refcheck=False)
    _fmap_.resize(_meta_["edges"], refcheck=False)
    _equalPred_.resize(_meta_["edges"], refcheck=False)
    load_factors(factor_data, _meta_["factors"], _factors_, _fstart_, _fmap_, _equalPred_)
    #factor.byteswap() # TODO: only if system is little-endian
    # TODO: byteswap fstart, fmap, equalPredicate
    if print_info and not print_only_meta:
        print(_factors_)

    ## INITIALIZE VARIABLE MAPS ##
    _vstart_.resize(_meta_["variables"] + 1, refcheck=False)
    _vmap_.resize(_meta_["edges"], refcheck=False)
    compute_var_map(_fstart_, _fmap_, _vstart_, _vmap_)
    

#### MAIN METHOD ####

def main(argv=None,def_param=(_meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Runs a Gibbs sampler",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default=".",
                        type=str)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file") # TODO: print default for meta, weight, variable, factor in help
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    # TODO: burn-in option
    # TODO: learning options
    # TODO: inference option
    parser.add_argument("-l", "--learn",
                        metavar="NUM_LEARN_STEPS",
                        dest="learn",
                        default=0,
                        type=int,
                        help="number of learning sweeps")
    parser.add_argument("-i", "--inference",
                        metavar="NUM_INFERENCE_STEPS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference sweeps")
    parser.add_argument("-t", "--threads",
                        metavar="NUM_THREADS",
                        dest="threads",
                        default=1,
                        type=int,
                        help="number of threads")
    # TODO: sample observed variable option
    parser.add_argument("-q", "--quiet",
                        #metavar="QUIET",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        #type=bool,
                        help="quiet")
    # TODO: verbose option (print all info)
    parser.add_argument("--verbose",
    #                    metavar="VERBOSE",
                        dest="verbose",
                        default=False,
                        action="store_true",
    #                    type=bool,
                        help="verbose")
    parser.add_argument("--version",
                        action='version',
                        version="%(prog)s 0.0",
                        help="print version number")

    print(argv)
    arg = parser.parse_args(argv)
    print(arg)

    # Call load method for data loading
    load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, not arg.quiet, not arg.verbose)
    # Initialize sampler 
    var_copies = 1
    weight_copies = 1
    samplerInit()
    learn(arg.learn, 0.0001, 0, 0, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
    gibbs(arg.threads, arg.inference, 0, 0, _meta_, _weights_, _variables_, _factors_, _fstart_, _fmap_, _vstart_, _vmap_, _equalPred_, _Z_, _count_, _varCopies_, _weightCopies_)
    print(_count_[:1000])


if __name__ == "__main__":
    main()  
