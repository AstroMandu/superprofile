from numba import njit
import numpy as np

@njit(fastmath=True, cache=True, inline='always')
def sigmoid01_exp(z):            # faster than tanh form for |z|<=8
    return 1.0 / (1.0 + np.exp(-z))

@njit(fastmath=True, cache=True, inline='always')
def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

@njit(fastmath=True, cache=True, inline='always')
def inv_softplus(y):
    return y + np.log1p(-np.exp(-y))