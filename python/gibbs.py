#!/usr/bin/env python

import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import struct
import math
import random
import sys

Meta = np.dtype([('weights',        np.int64),
                 ('variables',      np.int64),
                 ('factors',        np.int64),
                 ('edges',          np.int64),
                 ('weights_file',   object), # TODO: no max length on string
                 ('variables_file', object),
                 ('factors_file',   object),
                 ('edges_file',     object)])
#Meta_ = numba.from_dtype(Meta)

# TODO: uint or int
Weight  = np.dtype([("weightId",     np.int64),
                    ("isFixed",      np.bool),
                    ("initialValue", np.float64)])
#Weight = Weight.newbyteorder("b") # TODO: This kills numba...
Weight_ = numba.from_dtype(Weight)

Variable  = np.dtype([("variableId",     np.int64),
                      ("roleSerialized", np.bool),
                      ("initialValue",   np.int32),
                      ("dataType",       np.int16),
                      ("cardinality",    np.int32)])
Variable_ = numba.from_dtype(Variable)

#VariableReference  = np.dtype([("variableId",     np.int64),
#                               ("equalPredicate", np.int32)])
#VariableReference_ = numba.from_dtype(VariableReference)
#
#WeightReference  = np.dtype([("weightId",     np.int64),
#                             ("featureValue", np.float64)])
#WeightReference_ = numba.from_dtype(WeightReference)

MAX_ARITY = 10 # TODO: need to make adaptable arrays
Factor  = np.dtype([("factorFunction", np.int16),
                    ("arity",          np.int32),
                    ("variableId",     np.int64, MAX_ARITY),
                    ("equalPredicate", np.int32, MAX_ARITY),
                    ("weightId",       np.int64),
                    ("featureValue",   np.float64)])
Factor_ = numba.from_dtype(Factor)

#Factor = Variable.newbyteorder(">")


#spec = [('weight', Weight_),
#        ('variable', Variable_)]
spec = [
        #('meta', Meta_[:]), # causes problems
        ('weight', Weight_[:]),
        ('variable', Variable_[:]),
        ('factor', Factor_[:]),
        ('vstart', numba.int64[:]),
        ('vmap',   numba.int64[:])
       ]

@jitclass(spec)
class FactorGraph(object):
    #class Variable(object):
    #    pass
    #
    #class Factor(object):
    #    pass
    #
    #class Weight(object):
    #    pass

    def __init__(self, weight, variable, factor, vstart, vmap):
        self.weight = weight
        self.variable = variable
        self.factor = factor
        self.vstart = vstart
        self.vmap = vmap

    def gibbs(self, sweeps):
        # TODO: give option do not store result, or just store tally
        sample = np.zeros((sweeps, self.variable.shape[0]), np.float64)
        #sample = np.zeros(sweeps, np.float64)
        for s in range(sweeps):
            for v in range(self.variable.shape[0]):
                sample[s, v] = self.sample(v)
                #sample[s] = self.sample(v)
        #print(sample)
        return sample

    def sample(self, var):
        Z = np.zeros(self.variable[var]["cardinality"])
        for i in range(self.variable[var]["cardinality"]):
            Z[i] = math.exp(self.potential(var, i))
        Z = np.cumsum(Z)
        z = random.random() * Z[-1]
        self.variable[var]["initialValue"] = np.argmax(Z >= z)
        return self.variable[var]["initialValue"]

    def potential(self, var, value):
        p = 0.0
        for i in range(self.vstart[self.variable[var]["variableId"]], self.vstart[self.variable[var]["variableId"] + 1]):
            p += self.eval_factor(self.vmap[i], var, value) # TODO: account for factor and weight
        return p

    #FUNC_IMPLY_NATURAL = 0,
    #FUNC_OR = 1,
    #FUNC_AND = 2,
    #FUNC_EQUAL = 3,
    def FUNC_ISTRUE(self, factor_id, var_id, value):
        factor = self.factor[factor_id]
        var = self.variable[var_id]
        for i in range(factor["arity"]):
            v = value if (factor["variableId"][i] == var["variableId"]) else self.variable[factor["variableId"][i]]["initialValue"]
            if v == 0:
                return 0
        return 1
    #FUNC_LINEAR = 7,
    #FUNC_RATIO = 8,
    #FUNC_LOGICAL = 9,
    #FUNC_AND_CATEGORICAL = 12,
    #FUNC_IMPLY_MLN = 13,

    def FUNC_UNDEFINED(self, factor, var, value):
        return 1
        #raise NotImplementedError("Function " + str(factor["factorFunction"]) + " is not implemented.") # TODO: make numba behave reasonably here

    
    def eval_factor(self, factor_id, var=-1, value=-1):
        if self.factor[factor_id]["factorFunction"] == 4:
            return self.FUNC_ISTRUE(factor_id, var, value)
        else:
            print("DEBUG****************************************************************************************************")
            return 1
            #raise NotImplementedError("Function " + str(factor["factorFunction"]) + " is not implemented.") # TODO: make numba behave reasonably here
            
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

        

def load():
    # TODO: check that entire file is read (nothing more/nothing less)
    meta = np.genfromtxt("../test/biased_coin/graph.meta",
                         delimiter=',',
                         dtype=Meta)
    print(meta)
    
    #weighs = np.empty(meta["variables"], Weight)
    
    weight = np.fromfile("../test/biased_coin/graph.weights", Weight).byteswap() # TODO: only if system is little-endian
    variable = np.fromfile("../test/biased_coin/graph.variables", Variable).byteswap() # TODO: only if system is little-endian
    print(weight)
    print(variable)
    factor = np.empty(meta["factors"], Factor)
    info = []
    with open("../test/biased_coin/graph.factors", "rb") as f: # TODO: should be weights_file...
        try:
            for i in range(meta["factors"]):
                factor[i]["factorFunction"] = struct.unpack('!h', f.read(2))[0]
                factor[i]["arity"] = struct.unpack('!i', f.read(4))[0]
                assert(factor[i]["arity"] <= MAX_ARITY)
                for j in range(factor[i]["arity"]):
                    factor[i]["variableId"][j] = struct.unpack('!q', f.read(8))[0]
                    factor[i]["equalPredicate"][j] = struct.unpack('!i', f.read(4))[0]
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
    print(factor)
    return meta, weight, variable, factor
    #    variable = np.empty(100, Variable)
    #    #variable["value"].value = 1
    #    #variable[1].value = 1
    #    #variable[1] = Variable

@jit
def compute_var_map(nvar, nedge, factor):
    vstart = np.zeros(nvar + 1, np.int64)
    vmap = np.zeros(nedge, np.int64)
  
    for f in factor:
        for i in range(f["arity"]):
            vstart[f["variableId"][i] + 1] += 1
  
    vstart = np.cumsum(vstart)
    index = vstart.copy()

    for (fi, f) in enumerate(factor):
        for i in range(f["arity"]):
            vmap[index[f["variableId"][i]]] = fi
            index[f["variableId"][i]] += 1
  
    return vstart, vmap

def main(argv=None):
    if argv is None:
        argv = sys.argv
    #fg = FactorGraph("../test/biased_coin/graph.meta")
    (meta, weight, variable, factor) = load()
    (vstart, vmap) = compute_var_map(meta["variables"], meta["edges"], factor)
    fg = FactorGraph(weight, variable, factor, vstart, vmap)
    fg.eval_factor(0, -1, -1)
    fg.potential(0, 1)
    fg.gibbs(100)

    print(fg.sample(0))

if __name__ == "__main__":
    main(sys.argv)

