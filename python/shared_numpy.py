#!/usr/bin/env python

import ctypes
import logging
import multiprocessing as mp

from contextlib import closing

import numpy as np

info = mp.get_logger().info

def main():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    # create shared array
    N, M = 100, 11
    shared_arr = mp.Array(ctypes.c_double, N)
    arr = tonumpyarray(shared_arr)

    # fill with random values
    arr[:] = np.random.uniform(size=N)
    arr_orig = arr.copy()

    # write to arr from different processes
    with closing(mp.Pool(initializer=init, initargs=(shared_arr,))) as p:
        # many processes access the same slice
        stop_f = N // 10
        p.map_async(f, [slice(stop_f)]*M)

        # many processes access different slices of the same array
        assert M % 2 # odd
        step = N // 10
        p.map_async(g, [slice(i, i + step) for i in range(stop_f, N, step)])
    p.join()
    assert np.allclose(((-1)**M)*tonumpyarray(shared_arr), arr_orig)

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
    arr[i] = -1 * arr[i]
    info("end   %s" % (i,))

if __name__ == '__main__':
    mp.freeze_support()
    main()

