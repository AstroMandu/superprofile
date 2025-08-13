
import numpy as np
from scipy.stats import f, gaussian_kde
from numba import njit

@njit
def gauss(x, amp, mu, sig):
    val = amp * np.exp(-0.5*np.square((x-mu)/sig))
    return val

@njit
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
    
    
@njit
def model_1G(xx,A1,V1,S1,B1):
    model = gauss(xx, A1,V1,S1)+B1
    return model

@njit
def model_2G(xx,A21,A22,V21,V22,S21,S22,B2):
    return gauss(xx, A21,V21,S21) + gauss(xx, A22,V22,S22) + B2

@njit
def linmap(param, low, div):
    return (param-low) * div

@njit
def chisq_gauss2(y,model,inv_e_y):
    return -0.5 * np.sum(np.square((y - model) * inv_e_y))

@njit
def test_off_bounds(mapped):
    for val in mapped:
        if val<0.0 or val>1.0:
            return True
    return False

@njit
def bound_to_div(bound):
    return 1/(bound[1]-bound[0])

@njit
def log_L_1G_jit(x, y, inv_e_y, A1, V1, S1, B1):
    model = gauss(x, A1, V1, S1) + B1
    return chisq_gauss2(y, model, inv_e_y)

@njit
def log_L_2G_jit(x, y, inv_e_y, A21, A22, V21, V22, S21, S22, B2):
    model = gauss(x, A21, V21, S21) + gauss(x, A22, V22, S22) + B2
    return chisq_gauss2(y, model, inv_e_y)

@njit
def _softplus(x):
    return np.log(1+np.exp(x))

@njit
def _inv_softplus(y):
    return np.log(np.exp(y)-1)

def _idx(names, key):
    idx = np.where(names == key)[0]
    return int(idx[0]) if idx.size > 0 else None