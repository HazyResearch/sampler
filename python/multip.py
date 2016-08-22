#!/usr/bin/env python

from __future__ import print_function
import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
import ctypes
import time

N = 10000
ITER = 1000000

S = np.zeros(N)
shape = S.shape
#S.shape = size
shared_array = sharedctypes.RawArray('d', S)
S = np.frombuffer(shared_array, dtype=np.float64, count=N)
S.shape = shape
#a = np.zeros(10000)
#shared_array[:10000] = a

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def f_wrapper((iters,array)):
    f(iters, array)
    print("Range:", min(array), "-", max(array))

@jit(nopython=True,nogil=True)
def f(iters,array):
    for it in range(iters):
        for i in range(len(array)):
            array[i] += 1


with Timer() as t:
    f_wrapper((ITER, S))
print('Request took %.03f sec.' % t.interval)
print("Range:", min(shared_array), "-", max(shared_array))
#shared_array[:10000] = a
with Timer() as t:
    pool = multiprocessing.Pool(processes=2)
    pool.map(f_wrapper,[(ITER,S)]*2)

print('Request took %.03f sec.' % t.interval)
print("Range:", min(S), "-", max(S))

