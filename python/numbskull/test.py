import numbskull
args = ['../../ising', '-l','100','-i', '200', '-t', '10','-s','0.001']
ns = numbskull.main(args)
ns.loadFGFromFile()
ns.learning()
ns.inference()

