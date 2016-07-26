#!/usr/bin/env python

import sys
import gibbs

def main(argv=None):
    if argv is None:
        argv = sys.argv
    #fg = gibbs.FactorGraph("../test/biased_coin/graph.meta")
    (meta, weight, variable, factor) = gibbs.load()
    (vstart, vmap) = gibbs.compute_var_map(meta["variables"], meta["edges"], factor)
    fg = gibbs.FactorGraph(weight, variable, factor, vstart, vmap)
    fg.eval_factor(0, -1, -1)
    fg.potential(0, 1)
    fg.gibbs(100)

    print(fg.sample(0))

if __name__ == "__main__":
    main(sys.argv)

