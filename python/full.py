#!/usr/bin/env python

import sys

import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import struct

Meta = np.dtype([('weights',   np.int64),
                 ('variables', np.int64),
                 ('factors',   np.int64),
                 ('edges',     np.int64),
                 ('weights_file', np.str_, 100), # TODO: no max length on string
                 ('variables_file', np.str_, 100),
                 ('factors_file', np.str_, 100),
                 ('edges_file',   np.str_, 100)])
Meta_ = numba.from_dtype(Meta)

# TODO: uint or int
Weight  = np.dtype([("weightId",     np.int64),
                    ("isFixed",      np.bool),
                    ("initialValue", np.float64)])
#Weight = Weight.newbyteorder("b") # TODO: This kills numba...
Weight_ = numba.from_dtype(Weight)

Variable  = np.dtype([("variableId",     np.int64),
                      ("roleSerialized", np.bool),
                      ("initialValue", np.int32),
                      ("dataType",     np.int16),
                      ("cardinality",    np.int32)])
Variable_ = numba.from_dtype(Variable)

Factor  = np.dtype([("variableId",     np.int64),
                      ("roleSerialized", np.bool),
                      ("initialValue", np.int32),
                      ("dataType",     np.int16),
                      ("cardinality",    np.int32)])
Factor_ = numba.from_dtype(Factor)
Factor = Variable.newbyteorder(">")


#spec = [('weight', Weight_),
#        ('variable', Variable_)]
spec = [
        #('meta', Meta_[:]), # causes problems
        ('weight', Weight_[:]),
        ('variable', Variable_[:]),
        #('factor', Factor_[:])
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

    def __init__(self, weight, variable):#, factor):
        self.weight = weight
        self.variable = variable
        #self.factor = factor

def load():
    # TODO: check that entire file is read (nothing more/nothing less)
    meta = np.genfromtxt("../test/biased_coin/graph.meta",
                         delimiter=',',
                         dtype=Meta)
    print(meta)
    
    #weighs = np.empty(meta["variables"], Weight)
    
    weight = np.fromfile("../test/biased_coin/graph.weights", Weight).byteswap()
    variable = np.fromfile("../test/biased_coin/graph.variables", Variable).byteswap()
    print(weight)
    print(variable)
    factor = []
    with open("../test/biased_coin/graph.factors", "rb") as f: # TODO: should be weights_file...
        try:
            for i in range(meta["factors"]):
                factor_type = struct.unpack('!h', f.read(2))[0]
                arity = struct.unpack('!h', f.read(2))[0]
                f.read(30)
                variable_id = np.zeros(arity)
                should_equal_to = np.zeros(arity)
                for j in range(arity):
                    variable_id[j] = struct.unpack('!q', f.read(8))[0]
                    should_equal_to[j] = struct.unpack('!q', f.read(8))[0]
                #variableId1     long    8
                #isPositive1     bool    1
                #variableId2     long    8
                #isPositive2     bool    1
                factor += [(factor_type, arity, variable_id.tolist(), should_equal_to.tolist())]
        except:
            pass
        finally:
            pass
    
    print(factor)
    return meta, weight, variable, factor
    #    variable = np.empty(100, Variable)
    #    #variable["value"].value = 1
    #    #variable[1].value = 1
    #    #variable[1] = Variable

def main(argv=None):
    if argv is None:
        argv = sys.argv
    #fg = gibbs.FactorGraph("../test/biased_coin/graph.meta")
    (meta, weight, variable, factor) = load()
    fg = FactorGraph(weight, variable)

if __name__ == "__main__":
    main(sys.argv)

