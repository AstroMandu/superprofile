
import numpy as np
from scipy.stats import f, gaussian_kde
from numba import njit
import math

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

# tanh-based sigmoid on (−∞,∞) → (0,1)
@njit(fastmath=True, cache=True, inline='always')
def _sigmoid01_tanh(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)

# mapped sigmoid: (−∞,∞) → (lo, hi)
@njit(fastmath=True, cache=True)
def _sigmoid_mapped(raw, bound):
    lo = float(bound[0]); hi = float(bound[1])
    return lo + (hi - lo) * _sigmoid01_tanh(raw)

def _inv_sigmoid_mapped(scaled, bound, eps=1e-12):
    lo = float(bound[0]); hi = float(bound[1])
    span = hi - lo
    x = (scaled - lo) / span                 # (0,1)
    x = np.clip(x, eps, 1.0 - eps)           # keep away from endpoints
    # tanh inverse: z = 2 * atanh(2x - 1)
    y = 2.0 * x - 1.0                        # (-1,1)
    return 2.0 * np.arctanh(y)

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

def relabel_by_width(samples, names):
    """
    Relabel so S21 <= S22, swapping (A,V,S) consistently per draw.
    Accepts:
      - samples: (n_samples, ndim) or (ndim,)
      - names:   array-like with 'A21','A22','V21','V22','S21','S22'
    Returns same shape as input.
    """
    arr = np.asarray(samples)
    one_d = (arr.ndim == 1)
    work = arr[None, :].copy() if one_d else arr.copy()

    names = np.asarray(names)

    # required indices
    def idx(name):
        w = np.where(names == name)[0]
        if w.size == 0:
            raise KeyError(f"Parameter '{name}' not found in names.")
        return int(w[0])

    iS21, iS22 = idx('S21'), idx('S22')
    # optional pairs (in case someone runs without V’s, etc.)
    pairs = []
    for a,b in [('A21','A22'), ('V21','V22')]:
        try:
            pairs.append((idx(a), idx(b)))
        except KeyError:
            pass  # skip if missing

    # rows needing swap
    swap = work[:, iS21] > work[:, iS22]

    # vectorized column swap on masked rows
    def swap_cols(a, i, j, m):
        tmp = a[:, i].copy()
        a[m, i] = a[m, j]
        a[m, j] = tmp[m]

    # widths first (defines the ordering), then the paired params
    swap_cols(work, iS21, iS22, swap)
    for i, j in pairs:
        swap_cols(work, i, j, swap)

    return work[0] if one_d else work

def demap_params1G_unconstrained(params, gmodel):
    def demap_A1(A1): return _sigmoid_mapped(A1,gmodel.dict_bound['A1'])
    def demap_V1(V1): return _sigmoid_mapped(V1,gmodel.dict_bound['V1'])
    def demap_S1(S1): return _sigmoid_mapped(S1,gmodel.dict_bound['S1'])
    def demap_B1(B1): return _sigmoid_mapped(B1,gmodel.dict_bound['B1'])
    params_demapped = params.copy()
    if len(params.shape)>1:
        params_demapped[:,gmodel.iA1] = demap_A1(params[:,gmodel.iA1])
        params_demapped[:,gmodel.iV1] = demap_V1(params[:,gmodel.iV1])
        params_demapped[:,gmodel.iS1] = demap_S1(params[:,gmodel.iS1])
        params_demapped[:,gmodel.iB1] = demap_B1(params[:,gmodel.iB1])
    else:
        params_demapped[gmodel.iA1] = demap_A1(params[gmodel.iA1])
        params_demapped[gmodel.iV1] = demap_V1(params[gmodel.iV1])
        params_demapped[gmodel.iS1] = demap_S1(params[gmodel.iS1])
        params_demapped[gmodel.iB1] = demap_B1(params[gmodel.iB1])
    return params_demapped

def map_params1G_unconstrained(params, gmodel):
    def map_A1(A1): return _inv_sigmoid_mapped(A1,gmodel.dict_bound['A1'])
    def map_V1(V1): return _inv_sigmoid_mapped(V1,gmodel.dict_bound['V1'])
    def map_S1(S1): return _inv_sigmoid_mapped(S1,gmodel.dict_bound['S1'])
    def map_B1(B1): return _inv_sigmoid_mapped(B1,gmodel.dict_bound['B1'])
    params_mapped = params.copy()
    if len(params.shape)>1:
        params_mapped[:,gmodel.iA1] = map_A1(params[:,gmodel.iA1])
        params_mapped[:,gmodel.iV1] = map_V1(params[:,gmodel.iV1])
        params_mapped[:,gmodel.iS1] = map_S1(params[:,gmodel.iS1])
        params_mapped[:,gmodel.iB1] = map_B1(params[:,gmodel.iB1])
    else:
        params_mapped[gmodel.iA1] = map_A1(params[gmodel.iA1])
        params_mapped[gmodel.iV1] = map_V1(params[gmodel.iV1])
        params_mapped[gmodel.iS1] = map_S1(params[gmodel.iS1])
        params_mapped[gmodel.iB1] = map_B1(params[gmodel.iB1])
    return params_mapped

def demap_params2G_unconstrained(params, gmodel):
    def demap_A2X(A2X): return _sigmoid_mapped(A2X,gmodel.dict_bound['A21'])
    def demap_V2X(V2X): return _sigmoid_mapped(V2X,gmodel.dict_bound['V21'])
    def demap_S2X(S2X): return _sigmoid_mapped(S2X,gmodel.dict_bound['S21'])
    def demap_B2(B2):   return _sigmoid_mapped(B2, gmodel.dict_bound['B2'])
    params_demapped = params.copy()
    if len(params.shape)>1:
        params_demapped[:,gmodel.iA21] = demap_A2X(params[:,gmodel.iA21])
        params_demapped[:,gmodel.iA22] = demap_A2X(params[:,gmodel.iA22])
        params_demapped[:,gmodel.iS21] = demap_S2X(params[:,gmodel.iS21])
        params_demapped[:,gmodel.iS22] = demap_S2X(params[:,gmodel.iS22])
        if gmodel.has_V21: params_demapped[:,gmodel.iV21] = demap_V2X(params[:,gmodel.iV21])
        if gmodel.has_V22: params_demapped[:,gmodel.iV22] = demap_V2X(params[:,gmodel.iV22])
        if gmodel.has_B2:  params_demapped[:,gmodel.iB2]  = demap_B2( params[:,gmodel.iB2])
    else:
        params_demapped[gmodel.iA21] = demap_A2X(params[gmodel.iA21])
        params_demapped[gmodel.iA22] = demap_A2X(params[gmodel.iA22])
        params_demapped[gmodel.iS21] = demap_S2X(params[gmodel.iS21])
        params_demapped[gmodel.iS22] = demap_S2X(params[gmodel.iS22])
        if gmodel.has_V21: params_demapped[gmodel.iV21] = demap_V2X(params[gmodel.iV21])
        if gmodel.has_V22: params_demapped[gmodel.iV22] = demap_V2X(params[gmodel.iV22])
        if gmodel.has_B2:  params_demapped[gmodel.iB2]  = demap_B2( params[gmodel.iB2])
    return params_demapped

def map_params2G_unconstrained(params, gmodel):
    def map_A2X(A2X): return _inv_sigmoid_mapped(A2X,gmodel.dict_bound['A21'])
    def map_V2X(V2X): return _inv_sigmoid_mapped(V2X,gmodel.dict_bound['V21'])
    def map_S2X(S2X): return _inv_sigmoid_mapped(S2X,gmodel.dict_bound['S21'])
    def map_B2(B2):   return _inv_sigmoid_mapped(B2, gmodel.dict_bound['B2'])
    params_mapped = params.copy()
    if len(params.shape)>1:
        params_mapped[:,gmodel.iA21] = map_A2X(params[:,gmodel.iA21])
        params_mapped[:,gmodel.iA22] = map_A2X(params[:,gmodel.iA22])
        params_mapped[:,gmodel.iS21] = map_S2X(params[:,gmodel.iS21])
        params_mapped[:,gmodel.iS22] = map_S2X(params[:,gmodel.iS22])
        if gmodel.has_V21: params_mapped[:,gmodel.iV21] = map_V2X(params[:,gmodel.iV21])
        if gmodel.has_V22: params_mapped[:,gmodel.iV22] = map_V2X(params[:,gmodel.iV22])
        if gmodel.has_B2:  params_mapped[:,gmodel.iB2]  = map_B2( params[:,gmodel.iB2])
    else:
        params_mapped[gmodel.iA21] = map_A2X(params[gmodel.iA21])
        params_mapped[gmodel.iA22] = map_A2X(params[gmodel.iA22])
        params_mapped[gmodel.iS21] = map_S2X(params[gmodel.iS21])
        params_mapped[gmodel.iS22] = map_S2X(params[gmodel.iS22])
        if gmodel.has_V21: params_mapped[gmodel.iV21] = map_V2X(params[gmodel.iV21])
        if gmodel.has_V22: params_mapped[gmodel.iV22] = map_V2X(params[gmodel.iV22])
        if gmodel.has_B2:  params_mapped[gmodel.iB2]  = map_B2( params[gmodel.iB2])
    return params_mapped

@njit(fastmath=True, cache=True)
def _clip_raw(u, L=RAW_LIM):
    if u > L:   return L
    if u < -L:  return -L
    return u


def orthogonalize_rows(pos, magnitude=1e-2):
    # Center
    C = pos - pos.mean(axis=0, keepdims=True)
    nwalkers, ndim = C.shape
    # Build an orthonormal basis in R^(nwalkers) and project a small component
    Q, _ = np.linalg.qr(np.random.randn(nwalkers, nwalkers))
    bump = Q[:, :ndim] * magnitude  # nwalkers x ndim
    return pos + bump

@njit(fastmath=True, cache=True)
def check_sanity(params):
    for param in params:
        if param<-8 or param>8: return False
    return True


@njit(fastmath=True, cache=True)
def check_sanity_softplus(params):
    for param in params:
        if param<-20 or param>1e10: return False
    return True