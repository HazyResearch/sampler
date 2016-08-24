import numbskull
args = ['../../ising', '-l','100','-i', '100', '-t', '1','-s','0.001']
ns = numbskull.main(args)
ns.loadFGFromFile()
ns.learning()
ns.inference()
