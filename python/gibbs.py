import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import struct

Meta = np.dtype([('weights',   np.int),
                 ('variables', np.int),
                 ('factors',   np.int),
                 ('edges',     np.int),
                 ('weights_file', np.str_, 100), # TODO: no max length on string
                 ('variables_file', np.str_, 100),
                 ('factors_file', np.str_, 100),
                 ('edges_file',   np.str_, 100)])
Meta_ = numba.from_dtype(Meta)

# TODO: uint or int
Weight  = np.dtype([("weightId",     np.int64),
                    ("isFixed",      np.bool),
                    ("initialValue", np.float64)])
Weight_ = numba.from_dtype(Weight)
Weight = Weight.newbyteorder(">")

Variable  = np.dtype([("variableId",     np.int64),
                      ("roleSerialized", np.bool),
                      ("initialValue", np.int32),
                      ("dataType",     np.int16),
                      ("cardinality",    np.int32)])
Variable_ = numba.from_dtype(Variable)
Variable = Variable.newbyteorder(">")

#spec = [('weight', Weight_),
#        ('variable', Variable_)]
spec = [
        ('meta', Meta_),
        ('weights', Weight_[:])
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

    def __init__(self):
        # TODO: check that entire file is read (nothing more/nothing less)
        pass
        self.meta = np.genfromtxt("../test/biased_coin/graph.meta",
                             delimiter=',',
                             dtype=Meta)
        print(self.meta)

        #self.weighs = np.empty(meta["variables"], Weight)
        self.variable = np.empty(self.meta["variables"], Variable)

        self.weights = np.fromfile("../test/biased_coin/graph.weights", Weight)
        self.variables = np.fromfile("../test/biased_coin/graph.variables", Variable)
        print(self.weights)
        print(self.variables)
        info = []
        with open("../test/biased_coin/graph.factors", "rb") as f: # TODO: should be weights_file...
            try:
                for i in range(self.meta["factors"]):
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
                    info += [(factor_type, arity, variable_id.tolist(), should_equal_to.tolist())]
            except:
                pass
            finally:
                pass

        print(info)
        #    self.variable = np.empty(100, Variable)
        #    #self.variable["value"].value = 1
        #    #self.variable[1].value = 1
        #    #self.variable[1] = Variable

