#!/usr/bin/env python

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

Meta = np.dtype([('weights',        np.int64),
                 ('variables',      np.int64),
                 ('factors',        np.int64),
                 ('edges',          np.int64)])
                 #('weights_file',   object), # TODO: no max length on string
                 #('variables_file', object),
                 #('factors_file',   object),
                 #('edges_file',     object)])
#Meta_ = numba.from_dtype(Meta)

# TODO: uint or int
Weight  = np.dtype([("isFixed",      np.bool),
                    ("initialValue", np.float64)])
Weight_ = numba.from_dtype(Weight)
#Weight = Weight.newbyteorder("b") # TODO: This kills numba...

Variable  = np.dtype([("isEvidence",   np.bool),
                      ("initialValue", np.int32),
                      ("dataType",     np.int16),
                      ("cardinality",  np.int32)])
Variable_ = numba.from_dtype(Variable)

#VariableReference  = np.dtype([("variableId",     np.int64),
#                               ("equalPredicate", np.int32)])
#VariableReference_ = numba.from_dtype(VariableReference)
#
#WeightReference  = np.dtype([("weightId",     np.int64),
#                             ("featureValue", np.float64)])
#WeightReference_ = numba.from_dtype(WeightReference)

Factor  = np.dtype([("factorFunction", np.int16),
                    ("weightId",       np.int64),
                    ("featureValue",   np.float64),
                   ])
Factor_ = numba.from_dtype(Factor)

#Factor = Variable.newbyteorder(">")


#spec = [('weight', Weight_),
#        ('variable', Variable_)]
spec = [
        #('meta', Meta_[:]), # causes problems
        ('weight',         Weight_[:]),
        ('variable',       Variable_[:]),
        ('factor',         Factor_[:]),
        ('fstart',         numba.int64[:]),
        ('fmap',           numba.int64[:]),
        ('vstart',         numba.int64[:]),
        ('vmap',           numba.int64[:]),
        ('equalPredicate', numba.int32[:]),
        ('Z',              numba.float64[:])
       ]

@jitclass(spec)
class FactorGraph(object):

    def __init__(self, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate):
        self.weight = weight
        self.variable = variable
        self.factor = factor
        self.fstart = fstart
        self.fmap = fmap
        self.vstart = vstart
        self.vmap = vmap
        self.equalPredicate = equalPredicate

        cardinality = 0
        for v in self.variable:
            cardinality = max(cardinality, v["cardinality"])
        self.Z = np.zeros(cardinality)


    def gibbs(self, sweeps):
        # TODO: give option do not store result, or just store tally
        #sample = np.zeros((sweeps, self.variable.shape[0]), np.float64)
        sample = np.zeros(self.variable.shape[0], np.int32)
        #sample = np.zeros(sweeps, np.float64)
        for sweep in range(sweeps):
            for var_samp in range(self.variable.shape[0]):
                #sample[s, v] = self.sample(v)
                sample[var_samp] += self.sample(var_samp)
            print(sweep + 1)
            print(np.max(sample))
            print(np.min(sample))
            print()
        sample.sort()
        for i in range(0, self.variable.shape[0], self.variable.shape[0] / 100):
            print(i, sample[i])
        print()

        bins = 10
        count = np.zeros(bins, dtype=np.int64)
        for i in range(len(sample)):
            count[min(sample[i] * bins / sweeps, bins - 1)] += 1
        for i in range(bins):
            print(i, count[i])
        #print(count)
        # TODO: histogram
        #print(sample)
        #return sample

    def sample(self, var_samp):
        #print(var)
        cardinality = self.variable[var_samp]["cardinality"]
        for j in range(cardinality):
            self.Z[j] = math.exp(self.potential(var_samp, j))

        #Z = np.cumsum(Z)
        for j in range(1, cardinality):
            self.Z[j] += self.Z[j - 1]

        z = random.random() * self.Z[cardinality - 1]
        # TODO: I think this looks at the full vector, will be slow if one var has high cardinality
        self.variable[var_samp]["initialValue"] = np.argmax(self.Z >= z)
        #print(self.Z[0:cardinality])
        #print(z)
        #print(self.variable[var]["initialValue"])
        #print()

        return self.variable[var_samp]["initialValue"]

    def potential(self, var_samp, value):
        p = 0.0
        for k in range(self.vstart[var_samp], self.vstart[var_samp + 1]):
            factor_id = self.vmap[k]

            ### eval_factor ###
            ef = 1
            if self.factor[factor_id]["factorFunction"] == 3: # FUNC_EQUAL
                #print("FUNC_EQUAL")
                #print("var_id:", var_id)
                #print("value:", value)
                v = value if (self.fmap[self.fstart[factor_id]] == var_samp) else self.variable[self.fmap[self.fstart[factor_id]]]["initialValue"]
                #print(v)
                for l in range(self.fstart[factor_id] + 1, self.fstart[factor_id + 1]):
                    w = value if (self.fmap[l] == var_samp) else self.variable[self.fmap[l]]["initialValue"]
                    #print(w)
                    if v != w:
                        ef = -1
                        break
            elif self.factor[factor_id]["factorFunction"] == 4: # FUNC_ISTRUE
                for l in range(self.fstart[factor_id], self.fstart[factor_id + 1]):
                    v = value if (self.fmap[l] == var_samp) else self.variable[self.fmap[l]]["initialValue"]
                    if v == 0:
                        ef = -1
                        break
            else: # FUNC_UNDEFINED
                print("Error: Factor Function", self.factor[factor_id]["factorFunction"], "( used in factor", factor_id, ") is not implemented.")
                raise NotImplementedError("Factor function is not implemented.")
            ### end eval_factor ###

            #p += self.factor[self.vmap[i]]["featureValue"] \
            #   * self.weight[self.factor[self.vmap[i]]["weightId"]]["initialValue"] \
            #   * self.eval_factor(self.vmap[i], var_samp, value) # TODO: account for factor and weight
            p += self.weight[self.factor[self.vmap[k]]["weightId"]]["initialValue"] \
               * ef
        return p

    #FUNC_IMPLY_NATURAL = 0,
    #FUNC_OR = 1,
    #FUNC_AND = 2,
    #FUNC_LINEAR = 7,
    #FUNC_RATIO = 8,
    #FUNC_LOGICAL = 9,
    #FUNC_AND_CATEGORICAL = 12,
    #FUNC_IMPLY_MLN = 13,

    def eval_factor(self, factor_id, var_id=-1, value=-1):
        pass
            
        #return self.FUNC_UNDEFINED(self.factor[factor_id], var, value)
        #return {
        #    #FUNC_IMPLY_NATURAL = 0,
        #    #FUNC_OR = 1,
        #    #FUNC_AND = 2,
        #    #FUNC_EQUAL = 3,
        #    4: self.FUNC_ISTRUE,
        #    #FUNC_LINEAR = 7,
        #    #FUNC_RATIO = 8,
        #    #FUNC_LOGICAL = 9,
        #    #FUNC_AND_CATEGORICAL = 12,
        #    #FUNC_IMPLY_MLN = 13,
        #}.get(self.factor[factor_id]["factorFunction"], self.FUNC_UNDEFINED)(self.factor[factor_id], var, value)

        
def dataType(i):
  return {0: "Boolean",
          1: "Categorical"}.get(i, "Unknown")

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

@jit(nopython=True,cache=True)
def load_weights(data, nweights, weight):
    for i in range(nweights):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        reverse(data, 17 * i, 17 * i + 8)
        weightId = np.frombuffer(data[(17 * i):(17 * i + 8)], dtype=np.int64)[0]
        #weightId = np.frombuffer(data[(17 * i):(17 * i + 8):-1], dtype=np.int64)[0]
        isFixed      =               data[17 * i + 8]
        reverse(data, 17 * i + 9, 17 * i + 17)
        initialValue = np.frombuffer(data[(17 * i + 9):(17 * i + 17)], dtype=np.float64)[0]

        weight[weightId]["isFixed"] = isFixed
        weight[weightId]["initialValue"] = initialValue
    print("DONE WITH WEIGHTS")

@jit(nopython=True,cache=True)
def load_variables(data, nvariables, variable):
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
        #print(variableId)
        #print(isEvidence)
        #print(initialValue)
        #print(dataType)
        #print(cardinality)

        variable[variableId]["isEvidence"] = isEvidence
        variable[variableId]["initialValue"] = initialValue
        variable[variableId]["dataType"] = dataType
        variable[variableId]["cardinality"] = cardinality
    print("DONE WITH VARS")

@jit(nopython=True,cache=True)
def load_factors(data, nfactors, factor, fstart, fmap, equalPredicate):
    index = 0
    for i in range(nfactors):
        reverse(data, index, index + 2)
        factor[i]["factorFunction"] = np.frombuffer(data[index:(index + 2)], dtype=np.int16)[0]

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
        factor[i]["weightId"]     = np.frombuffer(data[index:(index + 8)], dtype=np.int64)[0]
        reverse(data, index + 8, index + 16)
        factor[i]["featureValue"] = np.frombuffer(data[(index + 8):(index + 16)], dtype=np.float64)[0]
        index += 16
        #variableId1     long    8
        #isPositive1     bool    1
        #variableId2     long    8
        #isPositive2     bool    1
        #info += [(factor_type, arity, variable_id.tolist(), equal_predicate.tolist(), weight_id, feature_value)]
    print("DONE WITH FACTORS")

def load(directory=".",
         metafile="graph.meta",
         weightfile="graph.weights",
         variablefile="graph.variables",
         factorfile="graph.factors",
         print_info=False,
         print_only_meta=False):

    # TODO: check that entire file is read (nothing more/nothing less)
    # TODO: make error when file does not exist less dumb
    meta = np.loadtxt(directory + "/" + metafile,
                      delimiter=',',
                      dtype=Meta)
    meta = meta[()] # convert from 0-dimensional ndarray to scalar
    if print_info:
        print("Meta:")
        print("    weights:  ", meta["weights"])
        print("    variables:", meta["variables"])
        print("    factors:  ", meta["factors"])
        print("    edges:    ", meta["edges"])
        print()
    
    weight_data = np.memmap(directory + "/" + weightfile, mode="c")
    weight = np.empty(meta["weights"], Weight)
    load_weights(weight_data, meta["weights"], weight)
    #weight.byteswap() # TODO: only if system is little-endian
    #del data # TODO: data.close()?
    if print_info and not print_only_meta:
        print("Weights:")
        for (i, w) in enumerate(weight):
            print("    weightId:", i)
            print("        isFixed:     ", w["isFixed"])
            print("        initialValue:", w["initialValue"])
        print()

    variable_data = np.memmap(directory + "/" + variablefile, mode="c")
    variable = np.empty(meta["variables"], Variable)
    load_variables(variable_data, meta["variables"], variable)
    #variable.byteswap() # TODO: only if system is little-endian
    # TODO: clear variable data?
    if print_info and not print_only_meta:
        print("Variables:")
        for (i, v) in enumerate(variable):
            print("    variableId:", i)
            print("        isEvidence:  ", v["isEvidence"])
            print("        initialValue:", v["initialValue"])
            print("        dataType:    ", v["dataType"], "(", dataType(v["dataType"]), ")")
            print("        cardinality: ", v["cardinality"])
            # TODO: print connected factors and num factors
            print()

    # TODO: might need to sort by factorId? (or just load in right spot)
    factor_data = np.memmap(directory + "/" + factorfile, mode="c")
    factor = np.empty(meta["factors"], Factor)
    fstart = np.zeros(meta["factors"] + 1, np.int64)
    fmap = np.zeros(meta["edges"], np.int64)
    equalPredicate = np.zeros(meta["edges"], np.int32) 
    load_factors(factor_data, meta["factors"], factor, fstart, fmap, equalPredicate)
    #factor.byteswap() # TODO: only if system is little-endian
    # TODO: byteswap fstart, fmap, equalPredicate
    if print_info and not print_only_meta:
        print(factor)

    vstart = np.zeros(meta["variables"] + 1, np.int64)
    vmap = np.zeros(meta["edges"], np.int64)
    compute_var_map(fstart, fmap, vstart, vmap)
    return meta, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate
    #    variable = np.empty(100, Variable)
    #    #variable["value"].value = 1
    #    #variable[1].value = 1
    #    #variable[1] = Variable



def print_factor(f):
    print("factorFunction: " + str(f["factorFunction"]))
    print("weightId:       " + str(f["weightId"]))
    print("featureValue:   " + str(f["featureValue"]))

def main(argv=None):
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
    parser.add_argument("-i", "--inference",
                        metavar="NUM_INFERENCE_STEPS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference sweeps")
    # TODO: sample observed variable option
    # TODO: quiet option
    # TODO: verbose option (print all info)
    parser.add_argument("--version",
                        action='version',
                        version="%(prog)s 0.0",
                        help="print version number")

    print(argv)
    arg = parser.parse_args(argv)
    print(arg)

    (meta, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate) = load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, True, True)
    #(meta, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate) = load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, True)

    fg = FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate)

    res = fg.gibbs(arg.inference)


if __name__ == "__main__":
    main()

# TODO: debug statement for factor
# TODO: print list of factors for vars and
#       print list of vars for factors

