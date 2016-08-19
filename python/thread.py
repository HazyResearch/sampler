#!/usr/bin/env python

from __future__ import print_function
import numba
from numba import jit, jitclass, autojit, void
import numpy as np
import threading
import time

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

@jit(nopython=True,nogil=True)
def f(array, a2, iters):
    for it in range(iters):
        for i in range(len(array)):
            array[i] += 1
            #array[i] += a2[i]
            #a2[i] = 1

N = 10000
ITER = 1000000

array = np.zeros(N)
a2 = np.ones(N)
with Timer() as t:
    f(array, a2, 1000000)
print('Request took %.03f sec.' % t.interval)
print("Range:", min(array), "-", max(array))

#print(N / 2)

array = np.zeros(N)
with Timer() as t:
    t1 = threading.Thread(target=f, args=[array[:N/2+1], array[N/2:], ITER])
    t2 = threading.Thread(target=f, args=[array[N/2:], array[:N/2], ITER])
    t1.start()
    t2.start()
    t1.join()
    t2.join()
print('Request took %.03f sec.' % t.interval)
print("Range:", min(array), "-", max(array))

