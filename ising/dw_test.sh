#!/bin/bash
../dw gibbs --burn_in 0 -i 100 -l 0 -t 1 -c 1 -m graph.meta -w graph.weights -f graph.factors -v graph.variables
