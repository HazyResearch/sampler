#!/usr/bin/env python

import sys
import gibbs

def main(argv=None):
    if argv is None:
        argv = sys.argv
    #fg = gibbs.FactorGraph("../test/biased_coin/graph.meta")
    (meta, weight, variable, factor) = gibbs.load()
    fg = gibbs.FactorGraph(weight, variable)

if __name__ == "__main__":
    main(sys.argv)

