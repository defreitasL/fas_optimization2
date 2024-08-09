# cython: boundscheck=False
# cython -a -c=-O3 -c=-march=native -c=-ffast-math -c=-funroll-loops
from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt
import numpy as np
cimport numpy as np
cimport cython
from fast_optimization2.metrics import backtot, opt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def multi_obj_func2(list metrics):
    """
    Multi-objective function to get indices of metrics.
    """
    cdef list metrics_name_list
    cdef int i
    cdef list idx = []
    
    metrics_name_list, _ = backtot()
    print('**Using the following metrics:**')
    
    for metric in metrics:
        idx.append(metrics_name_list.index(metric))
    
    for i in idx:
        print(metrics_name_list[i])
    
    return idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=1] obj_func(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation, list indexes):
    """
    Objective function to compute likes based on evaluation and simulation data.
    """
    cdef list metrics_name_list, mask
    cdef int n = len(indexes)
    cdef np.ndarray[double, ndim=1] likes = np.empty(n)
    cdef int i, idx

    metrics_name_list, mask = backtot()
    
    for i in range(n):
        idx = indexes[i]
        if mask[idx]:
            likes[i] = opt(idx, evaluation, simulation)
        else:
            likes[i] = 1 - opt(idx, evaluation, simulation)
    
    return likes
