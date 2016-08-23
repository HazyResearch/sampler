import numpy as np
from inference import *
from learning import *

class FactorGraph(object):
	def __init__(self, weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate, var_copies, weight_copies, fid, workers):
		self.weight   	= weight
		self.variable 	= variable
		self.factor   	= factor
		self.fstart   	= fstart
		self.fmap     	= fmap
		self.vstart  	= vstart
		self.vmap     	= vmap
		self.equalPred = equalPredicate
		self.count	= np.zeros(self.variable.shape[0], np.int64)
		
		self.var_value  = np.tile(self.variable[:]['initialValue'],(var_copies,1))
		self.weight_value = np.tile(self.weight[:]['initialValue'],(weight_copies,1))
		
		self.Z = np.zeros(max(self.variable[:]['cardinality']))
		
		self.fid = fid
		assert(workers > 0)
		self.threads = workers

	def clear(self):
		self.count[:] = 0

	def burnIn(self, epochs, var_copy=0, weight_copy=0):
		print ("FACTOR "+str(self.fid)+": STARTED BURN-IN...")
		shardID, nshards = 0, 1
		gibbsthread(shardID, nshards, epochs, var_copy, weight_copy, self.weight, self.variable, self.factor, self.fstart, self.fmap, self.vstart, self.vmap, self.equalPred, self.Z, self.count, self.var_value, self.weight_value, True) # NUMBA-based method. Implemented in inference.py
		print ("FACTOR "+str(self.fid)+": DONE WITH BURN-IN")

	def diagnostics(self, epochs):
		bins = 10
	        hist = np.zeros(bins, dtype=np.int64)
		for i in range(len(self.count)):
		    hist[min(self.count[i] * bins / epochs, bins - 1)] += 1
		for i in range(bins):
		    print(i, hist[i])

	def diagnosticsLearning(self,weight_copy=0):
            	print("Weights:")
		for (i, w) in enumerate(self.weight):
			print("    weightId:", i)
			print("        isFixed:", w["isFixed"])
			print("        weight: ", self.weight_value[weight_copy][i])
		print()
	
	def inference(self,burnin_epochs, epochs, diagnostics=False, var_copy=0, weight_copy=0):
		
		## Burn-in
		if burnin_epochs > 0:
			self.burnIn(burnin_epochs)
		
		## Run inference
		print ("FACTOR "+str(self.fid)+": STARTED INFERENCE")
		activeThreads = []
		# Initialize and start all threads
		for threadID in range(self.threads):
			t = threading.Thread(target=gibbsthread, args=[threadID, self.threads, epochs, var_copy, weight_copy, self.weight, self.variable, self.factor, self.fstart, self.fmap, self.vstart, self.vmap, self.equalPred, self.Z, self.count, self.var_value, self.weight_value, False]) # NUMBA-based method. Implemented in inference.py
                        activeThreads.append(t)
                        t.start()
		# Wait for threads to finish
		for t in activeThreads:
			t.join()
		print ("FACTOR "+str(self.fid)+": DONE WITH INFERENCE")

		if diagnostics:
			self.diagnostics(epochs)

	def learn(self, burnin_epochs, epochs, stepsize, diagnostics=False, var_copy=0, weight_copy=0):
		# Burn-in
		if burnin_epochs > 0:
			self.burnIn(burnin_epochs)

		# Run learning
		print ("FACTOR "+str(self.fid)+": STARTED LEARNING")
                activeThreads = []
                # Initialize and start all threads
                for threadID in range(self.threads):
                        t = threading.Thread(target=learnthread, args=[threadID, self.threads, epochs, stepsize, var_copy, weight_copy, self.weight, self.variable, self.factor, self.fstart, self.fmap, self.vstart, self.vmap, self.equalPred, self.Z, self.count, self.var_value, self.weight_value]) # NUMBA-based method. Implemented in inference.py
                        activeThreads.append(t)
                        t.start()
                # Wait for threads to finish
                for t in activeThreads:
                        t.join()
                print ("FACTOR "+str(self.fid)+": DONE WITH LEARNING")

		if diagnostics:
			self.diagnosticsLearning(weight_copy)
