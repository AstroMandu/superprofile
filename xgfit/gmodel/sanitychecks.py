from numba import njit
import numpy as np

@njit(fastmath=True, cache=True, inline='always')
def check_sanity_3G_V32xV33xB3x(A31,A32,A33,V31,S31,S32,S33):
    if abs(A31)>8.0: return False
    if abs(A32)>8.0: return False
    if abs(A33)>8.0: return False
    if abs(V31)>8.0: return False
    if abs(S31)>8.0: return False
    if abs(S32)>8.0: return False
    if abs(S33)>20.: return False
    return True

@njit(fastmath=True, cache=True, inline='always')
def check_sanity(params):
    for p in params:
        if p < -8.0 or p > 8.0:
            return False
    return True