from numba import njit
from .gmodel_math import sigmoid01_exp, softplus
from .subroutines_Plotfitter import idx
from typing import Literal
import numpy as np


@njit(fastmath=True, cache=True, inline='always')
def map_params_1G(uA1,uV1,uS1,uB1,
                  lowA1,spnA1,
                  lowV1,spnV1,
                  lowS1,spnS1,
                  lowB1,spnB1):
    A1 = lowA1 + spnA1 * sigmoid01_exp(uA1)
    V1 = lowV1 + spnV1 * sigmoid01_exp(uV1)
    S1 = lowS1 + spnS1 * sigmoid01_exp(uS1)
    B1 = lowB1 + spnB1 * sigmoid01_exp(uB1)
    return A1,V1,S1,B1

@njit(fastmath=True, cache=True, inline='always')
def map_params_2G(uA21,uA22,uV21,uS21,uS22,
                  lowA,spnA,lowV,spnV,lowS21,spnS21,hihS22,deltav):
    A21 = lowA + spnA * sigmoid01_exp(uA21)
    A22 = lowA + spnA * sigmoid01_exp(uA22)
    V21 = lowV + spnV * sigmoid01_exp(uV21)
    
    S21 = lowS21 + spnS21 * sigmoid01_exp(uS21)
    lowS22 = S21+deltav
    spnS22 = hihS22-lowS22
    S22 = lowS22 + spnS22 * sigmoid01_exp(uS22)
    # S22 = S21+deltav + softplus(uS22)
    return A21,A22,V21,S21,S22

@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_A31A32A33V31(uA31,uA32,uA33,uV31,lowA,spnA,lowV,spnV):
    A31 = lowA + spnA * sigmoid01_exp(uA31)
    A32 = lowA + spnA * sigmoid01_exp(uA32)
    A33 = lowA + spnA * sigmoid01_exp(uA33)
    V31 = lowV + spnV * sigmoid01_exp(uV31)
    return A31,A32,A33,V31


@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_S31S32S33_sigmoid(uS31,uS32,uS33,lowS31,lowS32,spnS32,hihS33,deltav):
    S32   = lowS32 + spnS32 * sigmoid01_exp(uS32)
    
    spnS31 = S32-deltav-lowS31 
    S31   = lowS31 + spnS31 * sigmoid01_exp(uS31)
    
    lowS33 = S32+deltav
    spnS33 = hihS33-lowS33
    S33    = lowS33 + spnS33 * sigmoid01_exp(uS33)
    
    return S31,S32,S33

@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_S31S32S33_sigmoid2(uS31,uS32,uS33,
                                     lowS,spnS,lowS32,spnS32):
    
    S31 = lowS + spnS * sigmoid01_exp(uS31)
    S32 = lowS32 + spnS32 * sigmoid01_exp(uS32)
    S33 = lowS + spnS * sigmoid01_exp(uS33)
    return S31,S32,S33

@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_S31S32S33_sigmoidsoftplus(uS31,uS32,uS33,lowS31,lowS32,spnS32,deltav):
    S32   = lowS32 + spnS32 * sigmoid01_exp(uS32)
    
    spnS31 = S32-deltav-lowS31 
    S31    = lowS31 + spnS31 * sigmoid01_exp(uS31)
    
    S33   = S32+deltav+softplus(uS33)
    return S31,S32,S33

@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_S31S32S33_softplus(uS31,uS32,uS33,lowS31,deltav):
    S31 =     lowS31+softplus(uS31)
    S32 = S31+deltav+softplus(uS32)
    S33 = S32+deltav+softplus(uS33)
    return S31,S32,S33

@njit(fastmath=True, cache=True, inline='always')
def map_params_3G_A31A32A33V31V33(uA31,uA32,uA33,uV31,uV33,lowA,spnA,lowV31,spnV31,lowV33,spnV33):
    A31 = lowA + spnA * sigmoid01_exp(uA31)
    A32 = lowA + spnA * sigmoid01_exp(uA32)
    A33 = lowA + spnA * sigmoid01_exp(uA33)
    V31 = lowV31 + spnV31 * sigmoid01_exp(uV31)
    V33 = lowV33 + spnV33 * sigmoid01_exp(uV33)
    return A31,A32,A33,V31,V33


def _sigmoid_mapped(raw, low, high):
    # lo = float(bound[0]); hi = float(bound[1])
    return low + (high-low) * sigmoid01_exp(raw)

def _inv_sigmoid_mapped(scaled, low, high):
    x = (scaled-low)/(high-low)
    return np.log(x)-np.log1p(-x)

def _tanh_mapped(raw, cntr, hspan):
    return cntr + hspan * np.tanh(raw)

def _inv_tanh_mapped(scaled, cntr, hspan):
    x = (scaled - cntr)/hspan
    return np.arctanh(x)

def _linmap(scaled, low, spn):
    return low + spn * scaled

def _inv_linmap(raw, low, spn):
    return (raw-low)/spn


def _fwd_linear(u, low, spn): return low + spn*u
def _inv_linear(x, low, spn): return (x-low) / spn

def _fwd_logunif(u, loglow, logspn): return np.exp(loglow + logspn*u)
def _inv_logunif(x, loglow, logspn): return (np.log(x)-loglow)/logspn

def map_params(params, gmodel, mode:Literal['x->u','u->x']):
    
    low = gmodel.low
    hih = gmodel.hih
    spn = gmodel.spn
    dv  = gmodel.delta_v
    
    names = gmodel.names_param
    
    is_1d = (params.ndim==1)
    if is_1d: params = params[None,:]
    
    if mode=='u->x':
        mapper_linear  = _fwd_linear
        mapper_logunif = _fwd_logunif
        params_x = mapper_linear(params, low[None,:], spn[None,:])
        params_mapped = params_x
    elif mode=='x->u':
        mapper_linear  = _inv_linear
        mapper_logunif = _inv_logunif
        params_x = params
        params_mapped = mapper_linear(params, low[None,:], spn[None,:])
    else:
        raise
    
    #G1
    if 'S1' in names:
        iS1 = 2
        loglowS1,logspnS1 = np.log(low[None,iS1]),np.log(spn[None,iS1])
        params_mapped[:,iS1] = mapper_logunif(params[:,iS1],loglowS1,logspnS1)
    
    #G2
    elif 'S21' in names:
        iS21,iS22 = idx(names,['S21','S22'])
        S21    = params_x[:,iS21]
        loglowS22 = np.log(S21+dv)
        logspnS22 = np.log(hih[None,iS22])-loglowS22
        params_mapped[:,iS22] = mapper_logunif(params[:,iS22],loglowS22,logspnS22)
    
    elif 'S31' in names:
        iS31,iS32,iS33 = idx(names,['S31','S32','S33'])
        S32 = params_x[:,iS32]
        lowS31 = low[None,iS31]
        hihS31 = S32-dv
        spnS31 = hihS31-low[None,iS31]
        params_mapped[:,iS31] = mapper_linear(params[:,iS31], lowS31, spnS31)
        loglowS33 = np.log(S32+dv)
        logspnS33 = np.log(hih[None,iS33]) - loglowS33
        params_mapped[:,iS33] = mapper_logunif(params[:,iS33], loglowS33, logspnS33)
        
        # iF31,iF32,iF33 = idx(names,['F31','F32','F33'])
        # F32 = params_x[:,iF32]
        # lowF31,lowF33 = low[None,iF31],low[None,iF33]
        # params_mapped[:,iF31] = mapper_linear(params[:,iF31],lowF31,(F32-lowF31))
        # params_mapped[:,iF33] = mapper_linear(params[:,iF33],lowF33,(F32-lowF33))
        

    if is_1d: params_mapped = params_mapped[0]
            
    return params_mapped