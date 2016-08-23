import numbskull
args = ['../../ising', '-l','1000','-i', '1', '-t', '50','-s','0.00001']
ns = numbskull.main(args)
ns.loadFGFromFile()
ns.learning()
ns.inference()
