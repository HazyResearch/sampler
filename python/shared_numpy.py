#!/usr/bin/env python

from __future__ import print_function
import ctypes
import logging
import multiprocessing as mp
import time
from numba import jit

from contextlib import closing

import numpy as np

info = mp.get_logger().info

ITER = 10000

def main():
    #logger = mp.log_to_stderr()
    #logger.setLevel(logging.INFO)

    # create shared array
    N, M = 10000, 11
    shared_arr = mp.Array(ctypes.c_double, N)
    arr = tonumpyarray(shared_arr)

    # fill with random values
    arr[:] = np.zeros(N)
    #arr[:] = np.random.uniform(size=N)
    #arr_orig = arr.copy()

    start = time.time()
    h(arr)
    end = time.time()
    print("Time:", end - start)
    print("Range:", min(arr), "-", max(arr))
    print()

    # write to arr from different processes
    start = time.time()
    with closing(mp.Pool(initializer=init, initargs=(shared_arr,))) as p:
        ## many processes access the same slice
        #stop_f = N // 10
        #p.map_async(f, [slice(stop_f)]*M)

        # many processes access different slices of the same array
        #assert M % 2 # odd
        #step = N // 10
        p.map_async(g, [i for i in range(M)])
    p.join()
    end = time.time()
    print("Time:", end - start)
    print("Range:", min(arr), "-", max(arr))
    print()
    #assert np.allclose(((-1)**M)*tonumpyarray(shared_arr), arr_orig)

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_ # must be inhereted, not passed as an argument

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def f(i):
    """synchronized."""
    with shared_arr.get_lock(): # synchronize access
        g(i)

def g(i):
    """no synchronization."""
    info("start %s" % (i,))
    arr = tonumpyarray(shared_arr)
    h(arr)
    info("end   %s" % (i,))

@jit(nopython=True,cache=True,nogil=True)
def h(arr):
    for it in range(ITER):
        for i in range(len(arr)):
            arr[i] += 1

if __name__ == '__main__':
    mp.freeze_support()
    main()

