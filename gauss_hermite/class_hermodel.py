import warnings

from numba import njit
from herlogp import logP_njit
import numpy as np

class GHh3model:
    
    def __init__(self, xx, yy, inv_e_y):
        
        self.xx      = np.ascontiguousarray(xx)
        self.yy      = np.ascontiguousarray(yy)
        self.inv_e_y = np.ascontiguousarray(inv_e_y)
        
        self.dict_bounds = {}
    
    def make_logP(self):
        
        xx      = self.xx
        yy      = self.yy
        inv_e_y = self.inv_e_y
        
        lowA, spnA = self.dict_bounds['A']
        lowB, spnB = self.dict_bounds['B']
        lowZ, spnZ = self.dict_bounds['Z']
        
        loglowC, logspnC = self.dict_bounds['C']
        
        @njit(fastmath=True, cache=True, inline='always')
        def logP(ps):
            uA,uB,uC,h3,uZ = ps
            return logP_njit(xx,yy,inv_e_y,
                             uA,uB,uC,h3,uZ,
                             lowA,spnA,
                             lowB,spnB,
                             loglowC,logspnC,
                             lowZ,spnZ)
        
        return logP
    
    
