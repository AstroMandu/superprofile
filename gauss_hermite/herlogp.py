import math
from numba import njit
import numpy as np


@njit(fastmath=True, cache=True, inline='always')
def model(xx,A,B,invC,h3,Z):
    y = (xx-B)*invC
    return A*np.exp(-0.5*np.square(y)) * (1+h3/np.sqrt(6)*(2*np.sqrt(2)*np.power(y,3.)-3*np.sqrt(2)*y)) + Z

@njit(fastmath=True, cache=True, inline='always')
def logL_njit(xx,yy,inv_e_y,A,B,invC,h3,Z):
    acc = 0.0
    for i in range(xx.shape[0]):
        r = (yy[i] - model(xx[i],A,B,invC,h3,Z)) * inv_e_y[i]
        acc += r*r
    return 0.5 * acc


@njit(fastmath=True, cache=True, inline='always')
def logP_njit(xx,yy,inv_e_y,
              uA,uB,uC,h3,uZ,
              lowA,spnA,
              lowB,spnB,
              loglowC,logspnC,
              lowZ,spnZ):
    
    if (uA<0.0 or uA>1.0 or
        uB<0.0 or uB>1.0 or
        uC<0.0 or uC>1.0 or
        h3<-1.0 or h3>1.0 or
        uZ<0.0 or uZ>1.0): return np.inf
        
    A = lowA + spnA*uA
    B = lowB + spnB*uB
    C = math.exp(loglowC + logspnC*uC)  
    Z = lowZ + spnZ*uZ
    
    invC = 1.0/C
    
    logl =  logL_njit(xx,yy,inv_e_y,A,B,invC,h3,Z)
    return logl