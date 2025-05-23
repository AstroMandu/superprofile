import numpy as np
from numba import njit

@njit
def argfind_nearest(array, value):
    idx = np.searchsorted(array, value)
    idx = np.int64(idx)
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx-1
    else:
        return idx
    
@njit
def shifter(data, mod, lenx):
    
    lendata = data.shape[0]
    shifted = np.zeros(lenx, dtype=np.float64)
    shifted[mod:mod+lendata] = data
    
    return shifted
    