import math
from numba import njit

@njit(fastmath=True, cache=True, inline='always')
def log_L_1G_njit(x, y, inv_e_y, A1, V1, exp1, B1):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V1) * (x[i] - V1)
        r = (y[i] - B1 - A1 * math.exp(dx_sq * exp1)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc

@njit(fastmath=True, cache=True, inline='always')
def log_L_2G_njit(x, y, inv_e_y, A21, A22, V21, V22, exp1, exp2, B2):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq1 = (x[i] - V21) * (x[i] - V21)
        dx_sq2 = (x[i] - V22) * (x[i] - V22)
        r = (y[i] - B2
                   - A21 * math.exp(dx_sq1 * exp1)
                   - A22 * math.exp(dx_sq2 * exp2)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc


@njit(fastmath=True, cache=True, inline='always')
def log_L_2G_V22x_njit(x, y, inv_e_y, A21, A22, V2, exp1, exp2, B2):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V2) * (x[i] - V2)
        r = (y[i] - B2
                   - A21 * math.exp(dx_sq * exp1)
                   - A22 * math.exp(dx_sq * exp2)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc

@njit(fastmath=True, cache=True, inline='always')
def log_L_3G_V32xV33x_njit(x, y, inv_e_y, A31, A32, A33, V3, exp1, exp2, exp3, B3):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V3) * (x[i] - V3)
        r = (y[i] - B3
             - A31 * math.exp(dx_sq * exp1)
             - A32 * math.exp(dx_sq * exp2)
             - A33 * math.exp(dx_sq * exp3)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc
