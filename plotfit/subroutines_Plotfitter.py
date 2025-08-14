
import numpy as np
from scipy.stats import f, gaussian_kde
from numba import njit

S_EPS  = np.float64(1e-12)
S_MAX = np.float64(1e3) 
RAW_LIM = np.float64(40.0)

@njit(fastmath=True, cache=True)
def gauss(x, amp, mu, sig):
    val = amp * np.exp(-0.5*np.square((x-mu)/sig))
    return val

@njit(fastmath=True, cache=True)
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

def calc_red_chisq_1G(gmodel, df):
    
    xx = gmodel.x
    yy = gmodel.y
    e_y = gmodel.e_y
    
    A1, V1, S1, B1 = df.loc[0, ['A1', 'V1', 'S1', 'B1']]
    res_1G   = yy - (gauss(xx, A1,V1,S1)+B1)
    chisq_1G = np.sum((res_1G/e_y)**2) 
    dof_1G   = len(yy)-4
    
    return chisq_1G/dof_1G

def calc_red_chisq_2G(gmodel, df):
    
    xx = gmodel.x
    yy = gmodel.y
    e_y = gmodel.e_y
    
    A21, A22, V21, V22, S21, S22, B2 = df.loc[0, ['A21','A22','V21','V22','S21','S22','B2']]
    res_2G   = yy - (gauss(xx, A21,V21,S21)+gauss(xx, A22,V22,S22)+B2)
    chisq_2G = np.sum((res_2G/e_y)**2) 
    dof_2G   = len(yy) - len(gmodel.names_param) 
    
    return chisq_2G/dof_2G

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


def get_asymmetry_residuals(yy,residual):
    
    # asym = np.sum(np.abs(residual - np.flip(residual))) / (2*np.sum(np.abs(residual)))
    
    asym = np.sum(yy - np.flip(yy)) / (2*np.sum(yy))
    
    return asym
    
    
@njit(fastmath=True, cache=True)
def model_1G(xx,A1,V1,S1,B1):
    model = gauss(xx, A1,V1,S1)+B1
    return model

@njit(fastmath=True, cache=True)
def model_2G(xx,A21,A22,V21,V22,S21,S22,B2):
    return gauss(xx, A21,V21,S21) + gauss(xx, A22,V22,S22) + B2

@njit(fastmath=True, cache=True)
def linmap(param, low, div):
    return (param-low) * div

@njit(fastmath=True, cache=True)
def chisq_gauss2(y,model,inv_e_y):
    return -0.5 * np.sum(np.square((y - model) * inv_e_y))

@njit(fastmath=True, cache=True)
def test_off_bounds(mapped):
    for val in mapped:
        if val<0.0 or val>1.0:
            return True
    return False

@njit(fastmath=True, cache=True)
def bound_to_div(bound):
    return 1/(bound[1]-bound[0])

@njit(fastmath=True, cache=True)
def log_L_1G_jit(x, y, inv_e_y, A1, V1, S1, B1):
    model = gauss(x, A1, V1, S1) + B1
    return chisq_gauss2(y, model, inv_e_y)

@njit(fastmath=True, cache=True)
def log_L_2G_jit(x, y, inv_e_y, A21, A22, V21, V22, S21, S22, B2):
    model = gauss(x, A21, V21, S21) + gauss(x, A22, V22, S22) + B2
    return chisq_gauss2(y, model, inv_e_y)

@njit(fastmath=True, cache=True)
def _softplus(z, eps=S_EPS):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) + eps

@njit(fastmath=True, cache=True)
def _inv_softplus(y):
    return y + np.log1p(-np.exp(-y))

@njit(fastmath=True, cache=True)
def _sigmoid_mapped(raw, bound):
    """
    Map raw real values -> scaled in (lo, hi) using a logistic function.
    
    Parameters
    ----------
    raw : float or ndarray
        Unbounded input values.
    lo, hi : float
        Lower and upper bounds of the target range.
    """
    lo,hi = bound
    return lo + (hi - lo) / (1.0 + np.exp(-raw))

def _inv_sigmoid_mapped(scaled, bound, eps=1e-12):
    """
    Inverse of scaled_sigmoid: map scaled in (lo, hi) -> raw real values.
    
    Parameters
    ----------
    scaled : float or ndarray
        Values in (lo, hi) to map back to ℝ.
    lo, hi : float
        Same bounds as used in scaled_sigmoid.
    eps : float
        Small epsilon to keep away from exactly lo or hi to avoid infs.
    """
    lo,hi=bound
    # Normalize to (0, 1)
    x = (scaled - lo) / (hi - lo)
    # Clip to avoid log(0) or log(∞)
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))

@njit(fastmath=True, cache=True)
def _bounded_tanh(u, bound):
    a,b = bound
    return a + (b-a) * (np.tanh(u)*0.5+0.5)

def _inv_bounded_tanh(x, bound):
    a,b = bound
    # Clamp to avoid atanh hitting ±1 exactly due to float rounding
    z = 2 * (x - a) / (b - a) - 1
    z = np.clip(z, -1 + 1e-15, 1 - 1e-15)
    return np.arctanh(z)    

@njit(fastmath=True, cache=True)
def _softabs(u):
    return np.logaddexp(u,-u) - np.log(2.0)


def _inv_softabs(y):
    # y = log(cosh u) >= 0
    y = np.asarray(y)
    out = np.empty_like(y)

    # threshold where exp(y) starts to risk overflow in double
    y0 = np.float64(20.0)

    small = y <= y0
    large = ~small

    # small: exact formula u = arcosh(exp(y))
    if np.any(small):
        z = np.exp(y[small])
        out[small] = np.arccosh(z)

    # large: asymptotic u ≈ y + log 2 (since cosh u ~ e^u/2)
    if np.any(large):
        out[large] = y[large] + np.log(2.0)

    return out

def _idx(names, key):
    idx = np.where(names == key)[0]
    return int(idx[0]) if idx.size > 0 else None

def demap_params_unconstrained(params, gmodel):
    
    def demap_A21(A21): return _softplus(A21)
    def demap_A22(A22): return _softplus(A22)
    def demap_S21_S22(S21,S22):
        S21,S22 = _clip_raw(S21), _clip_raw(S22)
        uS21 = _softplus(S21) + S_EPS
        uS22 = _softabs(S22) + uS21 + S_EPS
        uS21 = np.clip(uS21, S_EPS, S_MAX)
        uS22 = np.clip(uS22, uS21 + S_EPS, S_MAX)
        return uS21, uS22
    names = gmodel.names_param
    iA21,iA22 = _idx(names,'A21'),_idx(names,'A22')
    iS21,iS22 = _idx(names,'S21'),_idx(names,'S22')
    params_demapped = params.copy()
    if len(params.shape)>1:
        params_demapped[:,iA21] = demap_A21(params[:,iA21])
        params_demapped[:,iA22] = demap_A22(params[:,iA22])
        params_demapped[:,iS21],params_demapped[:,iS22] = demap_S21_S22(params[:,iS21],params[:,iS22])
    else:
        params_demapped[iA21] = demap_A21(params[iA21])
        params_demapped[iA22] = demap_A21(params[iA22])
        params_demapped[iS21],params_demapped[iS22] = demap_S21_S22(params[iS21],params[iS22])
    
    return params_demapped

def map_params_unconstrained(params, gmodel):
    
    def map_A21(uA21): return _inv_softplus(uA21)
    def map_A22(uA22): return _inv_softplus(uA22)
    def map_S21_S22(uS21, uS22):
        uS21_eff = np.maximum(uS21 - S_EPS, S_EPS)
        d_eff    = np.maximum(uS22 - uS21 - S_EPS, 0.0)
        S21 = _inv_softplus(uS21_eff)
        S22 = _inv_softabs(d_eff)
        return S21, S22
    
    names = gmodel.names_param
    iA21,iA22 = _idx(names,'A21'),_idx(names,'A22')
    iS21,iS22 = _idx(names,'S21'),_idx(names,'S22')
    
    params_mapped = params.copy()
    
    if len(params.shape)>1:
        params_mapped[:,iA21] = map_A21(params[:,iA21])
        params_mapped[:,iA22] = map_A22(params[:,iA22])
        params_mapped[:,iS21],params_mapped[:,iS22] = map_S21_S22(params[:,iS21],params[:,iS22])
    else:
        params_mapped[iA21] = map_A21(params[iA21])
        params_mapped[iA22] = map_A21(params[iA22])
        params_mapped[iS21],params_mapped[iS22] = map_S21_S22(params[iS21],params[iS22])
    
    return params_mapped

@njit(fastmath=True, cache=True)
def _clip_raw(u, L=RAW_LIM):
    if u > L:   return L
    if u < -L:  return -L
    return u