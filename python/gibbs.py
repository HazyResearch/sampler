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
Weight  = np.dtype([("weightId",     np.int64),
                    ("isFixed",      np.bool),
                    ("initialValue", np.float64)])
#Weight = Weight.newbyteorder("b") # TODO: This kills numba...
Weight_ = numba.from_dtype(Weight)

Variable  = np.dtype([("variableId",   np.int64),
                      ("isEvidence",   np.bool),
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
                    ("arity",          np.int32),
                    ("weightId",       np.int64),
                    ("featureValue",   np.float64),
                   ])
Factor_ = numba.from_dtype(Factor)

#Factor = Variable.newbyteorder(">")


#spec = [('weight', Weight_),
#        ('variable', Variable_)]
spec = [
        #('meta', Meta_[:]), # causes problems
        ('weight', Weight_[:]),
        ('variable', Variable_[:]),
        ('factor', Factor_[:]),
        ('fstart', numba.int64[:]),
        ('fmap',   numba.int64[:]),
        ('vstart', numba.int64[:]),
        ('vmap',   numba.int64[:]),
        ('Z',      numba.float64[:])
       ]

@jitclass(spec)
class FactorGraph(object):

    def __init__(self, weight, variable, factor, fstart, fmap, vstart, vmap):
        self.weight = weight
        self.variable = variable
        self.factor = factor
        self.fstart = fstart
        self.fmap = fmap
        self.vstart = vstart
        self.vmap = vmap

        cardinality = 0
        for v in self.variable:
            cardinality = max(cardinality, v["cardinality"])
        self.Z = np.zeros(cardinality)


    def gibbs(self, sweeps):
        # TODO: give option do not store result, or just store tally
        #sample = np.zeros((sweeps, self.variable.shape[0]), np.float64)
        #sample = np.zeros(sweeps, np.float64)
        for s in range(sweeps):
            for v in range(self.variable.shape[0]):
                self.sample(v)
                #sample[s, v] = self.sample(v)
                #sample[s] = self.sample(v)
        #print(sample)
        #return sample

    def sample(self, var):
        cardinality = self.variable[var]["cardinality"]
        for i in range(cardinality):
            self.Z[i] = math.exp(self.potential(var, i))

        #Z = np.cumsum(Z)
        for i in range(1, cardinality):
            self.Z[i] += self.Z[i - 1]

        z = random.random() * self.Z[cardinality - 1]
        #print(Z)
        self.variable[var]["initialValue"] = np.argmax(self.Z >= z)
        return self.variable[var]["initialValue"]

    def potential(self, var, value):
        p = 0.0
        for i in range(self.vstart[var], self.vstart[var + 1]):
            #p += self.factor[self.vmap[i]]["featureValue"] \
            #   * self.weight[self.factor[self.vmap[i]]["weightId"]]["initialValue"] \
            #   * self.eval_factor(self.vmap[i], var, value) # TODO: account for factor and weight
            p += self.weight[self.factor[self.vmap[i]]["weightId"]]["initialValue"] \
               * self.eval_factor(self.vmap[i], var, value) # TODO: account for factor and weight
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
        if self.factor[factor_id]["factorFunction"] == 3: # FUNC_EQUAL
            v = value if (self.fmap[self.fstart[factor_id]] == var_id) else self.variable[self.fmap[self.fstart[factor_id]]]["initialValue"]
            for i in range(self.fstart[factor_id] + 1, self.fstart[factor_id + 1]):
                w = value if (self.fmap[i] == var_id) else self.variable[self.fmap[i]]["initialValue"]
                if v != w:
                    return 0
            return 1
        elif self.factor[factor_id]["factorFunction"] == 4: # FUNC_ISTRUE
            for i in range(self.fstart[factor_id], self.fstart[factor_id + 1]):
                v = value if (self.fmap[i] == var_id) else self.variable[self.fmap[i]]["initialValue"]
                if v == 0:
                    return 0
            return 1
        else: # FUNC_UNDEFINED
            print("Error: Factor Function", self.factor[factor_id]["factorFunction"], "( used in factor", factor_id, ") is not implemented.")
            raise NotImplementedError("Factor function is not implemented.")
            
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


def load(directory=".",
         metafile="graph.meta",
         weightfile="graph.weights",
         variablefile="graph.variables",
         factorfile="graph.factors",
         print_info=False,
         print_only_meta=False):

    # TODO: check that entire file is read (nothing more/nothing less)
    # TODO: make error when file does not exist less dumb
    meta = np.genfromtxt(directory + "/" + metafile,
                         delimiter=',',
                         dtype=Meta)
    if print_info:
        print("Meta:")
        print("    weights:  ", meta["weights"])
        print("    variables:", meta["variables"])
        print("    factors:  ", meta["factors"])
        print("    edges:    ", meta["edges"])
        print()
    
    #weighs = np.empty(meta["variables"], Weight)
    
    # TODO: need to sort by weightID
    weight = np.fromfile(directory + "/" + weightfile, Weight).byteswap() # TODO: only if system is little-endian
    if print_info and not print_only_meta:
        print("Weights:")
        for w in weight:
            print("    weightId:", w["weightId"])
            print("        isFixed:     ", w["isFixed"])
            print("        initialValue:", w["initialValue"])
        print()

    # TODO: need to sort by variableId
    variable = np.fromfile(directory + "/" + variablefile, Variable).byteswap() # TODO: only if system is little-endian
    if print_info and not print_only_meta:
        print("Variables:")
        for v in variable:
            print("    variableId:", v["variableId"])
            print("        isEvidence:  ", v["isEvidence"])
            print("        initialValue:", v["initialValue"])
            print("        dataType:    ", v["dataType"], "(", dataType(v["dataType"]), ")")
            print("        cardinality: ", v["cardinality"])
            # TODO: print connected factors and num factors
            print()

    # TODO: might need to sort by factorId? (or just load in right spot)
    factor = np.empty(meta["factors"], Factor)
    fstart = np.zeros(meta["factors"] + 1, np.int64)
    fmap = np.zeros(meta["edges"], np.int64)
    equalPredicate = np.zeros(meta["edges"], np.int32) 
    with open(directory + "/" + factorfile, "rb") as f:
        try:
            for i in range(meta["factors"]):
                factor[i]["factorFunction"] = struct.unpack('!h', f.read(2))[0]
                factor[i]["arity"] = struct.unpack('!i', f.read(4))[0]

                fstart[i + 1] = fstart[i] + factor[i]["arity"]
                for j in range(factor[i]["arity"]):
                    fmap[fstart[i] + j] = struct.unpack('!q', f.read(8))[0]
                    equalPredicate[fstart[i] + j] = struct.unpack('!i', f.read(4))[0]
                # TODO: handle FUNC_AND_CATEGORICAL
                factor[i]["weightId"]     = struct.unpack('!q', f.read(8))[0]
                factor[i]["featureValue"] = struct.unpack('!d', f.read(8))[0]
                #variableId1     long    8
                #isPositive1     bool    1
                #variableId2     long    8
                #isPositive2     bool    1
                #info += [(factor_type, arity, variable_id.tolist(), equal_predicate.tolist(), weight_id, feature_value)]
        finally:
            pass
    
    factor.byteswap() # TODO: only if system is little-endian
    if print_info and not print_only_meta:
        print(factor)
    return meta, weight, variable, factor, fstart, fmap
    #    variable = np.empty(100, Variable)
    #    #variable["value"].value = 1
    #    #variable[1].value = 1
    #    #variable[1] = Variable

@jit
def compute_var_map(nvar, nedge, fstart, fmap):
    vstart = np.zeros(nvar + 1, np.int64)
    vmap = np.zeros(nedge, np.int64)
  
    for i in fmap:
        vstart[i + 1] += 1
  
    vstart = np.cumsum(vstart)
    index = vstart.copy()

    for i in range(len(fstart) - 1):
        for j in range(fstart[i], fstart[i + 1]):
            vmap[index[fmap[j]]] = i
            index[fmap[j]] += 1
  
    return vstart, vmap

def print_factor(f):
    print("factorFunction: " + str(f["factorFunction"]))
    print("arity:          " + str(f["arity"]))
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

    (meta, weight, variable, factor, fstart, fmap) = load(arg.directory, arg.meta, arg.weight, arg.variable, arg.factor, True, True)
    (vstart, vmap) = compute_var_map(meta["variables"], meta["edges"], fstart, fmap)

    fg = FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap)

    res = fg.gibbs(arg.inference)


if __name__ == "__main__":
    main()

