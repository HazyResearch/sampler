#!/usr/bin/env python

from __future__ import print_function
import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import multiprocessing
import ctypes
import time

N = 10000
ITER = 10000

shared_array = multiprocessing.Array('f', 10000, lock=False)
a = np.zeros(10000)
shared_array[:10000] = a

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

#@jit(nopython=True,nogil=True)
@autojit
def f(iters,def_param=shared_array):
    for it in range(iters):
        for i in range(len(shared_array)):
            shared_array[i] += 1


a = np.zeros(10000)
with Timer() as t:
    f(ITER)
print('Request took %.03f sec.' % t.interval)
print("Range:", min(shared_array), "-", max(shared_array))
shared_array[:10000] = a
with Timer() as t:
    pool = multiprocessing.Pool(processes=10)
    pool.map(f,[ITER]*10)

print('Request took %.03f sec.' % t.interval)
print("Range:", min(shared_array), "-", max(shared_array))

