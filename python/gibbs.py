#!/usr/bin/env python

import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import struct

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
        ('factor', Factor_[:])
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

    def __init__(self, weight, variable, factor):
        self.weight = weight
        self.variable = variable
        self.factor = factor

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

