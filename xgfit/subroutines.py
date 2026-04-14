import math

import numpy as np
from numba import njit
from scipy.stats import f, gaussian_kde
from .gmodel.numeric import sigmoid01_exp, softplus, inv_softplus
import dcor

S_EPS  = np.float64(1e-12)
S_MAX = np.float64(1e3) 
RAW_LIM = np.float64(40.0)

def gauss(x, amp, mu, sig):
    inv = 1.0 / sig
    z = (x - mu) * inv
    return amp * np.exp(-0.5 * z * z)

def gaussian_area(amp, sigma):
    return amp*sigma*np.sqrt(2*np.pi)

def do_Ftest(gmodel, df, significance=0.05):
    
    xx = gmodel.x
    yy = gmodel.y
    e_y = gmodel.e_y
    
    # calculate 1Gres
    A1, V1, S1, B1 = df.loc[0, ['A1', 'V1', 'S1', 'B1']]
    res_1G   = yy - (gauss(xx, A1,V1,S1)+B1)
    chisq_1G = np.sum((res_1G/e_y)**2) 
    dof_1G   = len(yy)-4

    A21, A22, V21, V22, S21, S22, B2 = df.loc[0, ['A21','A22','V21','V22','S21','S22','B2']]
    
    res_2G   = yy - (gauss(xx, A21,V21,S21)+gauss(xx, A22,V22,S22)+B2)
    chisq_2G = np.sum((res_2G/e_y)**2) 
    dof_2G   = len(yy) - len(gmodel.names_param) 
    
    F = (chisq_1G - chisq_2G)/(dof_1G - dof_2G)/(chisq_2G/dof_2G)
    critical_value = f.ppf(1 - significance, dof_1G-dof_2G, dof_2G)
    
    return F, critical_value        


@njit(fastmath=True, cache=True)
def bound_to_div(bound):
    return 1/(bound[1]-bound[0])

@njit(fastmath=True, cache=True)
def bound_to_span(bound):
    return bound[1]-bound[0]


def get_mode(dist):
    kde = gaussian_kde(dist)
    grid = np.linspace(np.min(dist), np.max(dist), 1000)
    density = kde(grid)
    mode = density[np.argmax(density)]
    return mode


def sort_outliers(array, weight=1.5, return_index=False, only_higher=False):
    
    array = array[np.isfinite(array)]
    
    perc25, perc75 = np.nanpercentile(array, [25,75])
    IQR = perc75-perc25
    
    end_lower = perc25 - IQR*weight
    end_highr = perc75 + IQR*weight
    
    if only_higher:
        argwheres = np.argwhere(array<end_highr).flatten()
    else:
        argwheres = np.argwhere((array>end_lower) & (array<end_highr)).flatten()
    
    if return_index:
        return array[argwheres], argwheres
    return array[argwheres]


def idx(names, keys):
    arr = np.asarray(names, dtype=str)

    # scalar key
    if isinstance(keys, str):
        w = np.where(arr == keys)[0]
        return int(w[0]) if w.size > 0 else None

    # list/array of keys
    keys = np.asarray(keys, dtype=str)

    # ✅ if only one key, return scalar int (not array)
    if keys.size == 1:
        w = np.where(arr == keys.item())[0]
        return int(w[0]) if w.size > 0 else None

    out = np.full(len(keys), -1, dtype=int)
    for i, k in enumerate(keys):
        w = np.where(arr == k)[0]
        if w.size > 0:
            out[i] = int(w[0])
    return out


@njit(fastmath=True, cache=True, inline='always')
def makefit_bg_linefree_njit(x,y,e_y, V1,S1,B1, multiplier_S1=5):
   
    bound_low = V1 - multiplier_S1*S1
    bound_high= V1 + multiplier_S1*S1
    
    sum_wy = 0.0
    sum_w  = 0.0
    count  = 0 
    
    for i in range(x.size):
        xi = x[i]
        if xi<bound_low or xi>bound_high:
            wi = 1.0/(e_y[i]*e_y[i])
            sum_wy += wi * y[i]
            sum_w  += wi
            count  += 1
            
    if count<10:
        return B1
    
    BB = sum_wy / sum_w
    return BB if BB<B1 else B1


def clip_guess(guess, gmodel):
    _low = gmodel._low
    _hih = gmodel._hih
    _spn = gmodel._spn
    
    is_1d = (guess.ndim==1)
    if is_1d: guess = guess[None,:]
    
    if 'A31' in gmodel.names_param:
        iS31,iS32,iS33 = idx(gmodel.names_param,['S31','S32','S33'])
        _hihS31 = guess[:,iS32]-gmodel.delta_v
        _lowS33 = guess[:,iS32]+gmodel.delta_v
        _spnS31 = _hihS31 - _low[None,iS31]
        _spnS33 = _hih[None,iS33] - _lowS33
        _hih[None,iS31] = _hihS31
        _low[None,iS33] = _lowS33
        _spn[None,iS31] = _spnS31
        _spn[None,iS33] = _spnS33
    if 'A21' in gmodel.names_param:
        iS21,iS22 = idx(gmodel.names_param,['S21','S22'])
        _lowS22 = guess[:,iS21]+gmodel.delta_v
        _spnS22 = _hih[None,iS22] - _lowS22
        _low[None,iS22] = _lowS22
        _spn[None,iS22] = _spnS22
    
    guess = np.clip(guess, _low[None, :]+0.001*_spn[None, :], _hih[None, :]-0.001*_spn[None, :])
    
    if is_1d: guess = guess[0]
    
    return guess

# def dcor_trimmed(x,y, trim=5):
#     m = np.isfinite(x) & np.isfinite(y)

#     x = x[m]
#     y = y[m]

#     # --- percentile trimming (robust tail removal) ---
#     if trim > 0:
#         x_lo, x_hi = np.percentile(x, [trim, 100-trim])
#         y_lo, y_hi = np.percentile(y, [trim, 100-trim])

#         keep = (
#             (x >= x_lo) & (x <= x_hi) &
#             (y >= y_lo) & (y <= y_hi)
#         )
#         x = x[keep]
#         y = y[keep]
        
#     corr = dcor.distance_correlation(x,y)
#     return corr

def trim_outlier_mahalanobis(x,y,trim=5):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    
    data = np.column_stack([x, y])
    center = np.median(data, axis=0)       # robust center
    cov = np.cov(data.T)
    cov_inv = np.linalg.pinv(cov)
    
    diff = data - center
    # Mahalanobis distance squared for each point
    dist2 = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    
    threshold = np.percentile(dist2, 100 - trim)
    keep = dist2 <= threshold
    return x[keep], y[keep]


@njit(fastmath=True, cache=True, inline='always')
def get_interval_width(arr):
    n = len(arr)
    if n == 0: return 0.0
    
    idx16 = int(0.16 * n)
    idx84 = int(0.84 * n)
    
    # Numba supports np.partition as a function call.
    # Note: np.partition returns a partitioned copy in Numba.
    tmp = np.partition(arr, idx84)
    # Now we partition the first part again to find the 16th percentile
    tmp_lower = np.partition(tmp[:idx84], idx16)
    
    return tmp[idx84] - tmp_lower[idx16]

@njit(fastmath=True, cache=True)
def check_converged_resampling(samples, iters, nparams):
    idx_f = int(0.2 * iters)
    idx_l = int(0.8 * iters)
    
    for i in range(nparams):
        param_samples = samples[:iters, i]
        
        err_f = get_interval_width(param_samples[:idx_f])
        err_l = get_interval_width(param_samples[idx_l:])
        
        # Stability check
        if abs(err_f - err_l) / (err_l + 1e-10) > 0.05:
            return False
            
    return True


def compute_waic(log_liks):
    """
    log_likelihoods: array of shape (n_samples, n_data_points)
    Returns WAIC (lower = better)
    """
    # lppd: log pointwise predictive density
    lppd = np.sum(np.log(np.mean(np.exp(log_liks), axis=0)))
    
    # effective parameter count
    p_waic = np.sum(np.var(log_liks, axis=0))
    
    from scipy.special import logsumexp

    # Assuming log_liks is shape (n_samples, n_channels)
    print("log_liks shape:", log_liks.shape)
    print("any nan:", np.any(np.isnan(log_liks)))
    print("any -inf:", np.any(np.isneginf(log_liks)))
    print("any +inf:", np.any(np.isposinf(log_liks)))
    print("min:", np.nanmin(log_liks))
    print("max:", np.nanmax(log_liks))

    # Check lppd per channel before summing
    lppd_per_channel = logsumexp(log_liks, axis=0) - np.log(log_liks.shape[0])
    print("any -inf in lppd_per_channel:", np.any(np.isneginf(lppd_per_channel)))
    print("any +inf in lppd_per_channel:", np.any(np.isposinf(lppd_per_channel)))

    # Check p_waic per channel
    p_per_channel = np.var(log_liks, axis=0, ddof=1)
    print("any inf in p_per_channel:", np.any(~np.isfinite(p_per_channel)))
    print("max p_per_channel:", np.nanmax(p_per_channel))
    
    
    return -2 * (lppd - p_waic)


def compute_aic_bic_aicc(logl_max, k, n):
    aic  = -2*logl_max + 2*k
    bic  = -2*logl_max + k*np.log(n)
    aicc = aic + 2*k*(k+1)/(n - k - 1)
    return aic, bic, aicc