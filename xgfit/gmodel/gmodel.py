from math import isfinite
from pprint import pprint

import numpy as np
from numba import njit

from .logl_njit import (log_prob_1G_B1x_njit_linear,
                        log_prob_1G_njit_linear,
                        log_prob_2G_V22x_njit_linear,
                        log_prob_2G_V22xB2x_njit_linear,
                        log_prob_3G_V32xV33x_njit_linear,
                        log_prob_3G_V32xV33xB3x_njit_linear)
from .sanitychecks import check_sanity
from ..subroutines import bound_to_div, bound_to_span, idx


class Gmodel:

    def __init__(self, 
                 xx: np.ndarray, yy: np.ndarray, e_y: np.ndarray,
                 names_param, dict_bound,
                 df_plotfit=None):

        self.x       = np.ascontiguousarray(xx,  dtype=np.float64)
        self.y       = np.ascontiguousarray(yy,  dtype=np.float64)
        self.e_y     = np.ascontiguousarray(e_y, dtype=np.float64)
        self.inv_e_y = np.ascontiguousarray(1.0 / self.e_y, dtype=np.float64)
        
        self.delta_v = - np.abs(np.mean(np.diff(self.x))) / 2.355
        # self.delta_v = np.float64(0.0)
        
        self.low = np.float64([dict_bound[k][0]             for k in names_param])
        self.hih = np.float64([dict_bound[k][1]             for k in names_param])
        # self.div = np.float64([ bound_to_div(dict_bound[k]) for k in names_param])
        self.spn = np.float64([bound_to_span(dict_bound[k]) for k in names_param])
        
        self.BB = None

        self.df = df_plotfit
        self.names_param = names_param
        self.dict_bound  = dict_bound
    
    def update_bound(self, name_param, bound):
        bound = np.float64(bound)
        iP = idx(self.names_param,name_param)
        self.low[iP] = bound[0]
        # self.div[iP] = bound_to_div(bound)
        self.spn[iP] = bound_to_span(bound)
        self.dict_bound[name_param] = bound
    
    def make_log_prob_1G(self):
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y

        lowF1=self.low[0]; spnF1=self.spn[0]
        lowV1=self.low[1]; spnV1=self.spn[1]
        lowS1=self.low[2]; spnS1=self.spn[2]
        lowB1=self.low[3]; spnB1=self.spn[3]
        
        loglowS1 = np.log(lowS1)
        logspnS1 = np.log(spnS1)
        
        logprob_njit = log_prob_1G_njit_linear
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF1,uV1,uS1,uB1 = ps
            return logprob_njit(x,y,inv_e_y,
                                uF1,uV1,uS1,uB1,
                                lowF1,spnF1,
                                lowV1,spnV1,
                                loglowS1,logspnS1,
                                lowB1,spnB1)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps,y):
            uF1,uV1,uS1,uB1 = ps
            return -logprob_njit(x,y,inv_e_y,
                                uF1,uV1,uS1,uB1,
                                lowF1,spnF1,
                                lowV1,spnV1,
                                loglowS1,logspnS1,
                                lowB1,spnB1)
            
        return logprob, logprob_resample
    
    def make_log_prob_1G_B1x(self):
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y

        lowF1=self.low[0]; spnF1=self.spn[0]
        lowV1=self.low[1]; spnV1=self.spn[1]
        lowS1=self.low[2]; spnS1=self.spn[2]
        BB   =self.BB
        
        loglowS1 = np.log(lowS1)
        logspnS1 = np.log(spnS1)
        
        logprob_njit = log_prob_1G_B1x_njit_linear
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF1,uV1,uS1 = ps
            return logprob_njit(x,y,inv_e_y,
                                uF1,uV1,uS1,BB,
                                lowF1,spnF1,
                                lowV1,spnV1,
                                loglowS1,logspnS1)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps,y):
            uF1,uV1,uS1 = ps
            return -logprob_njit(x,y,inv_e_y,
                                 uF1,uV1,uS1,BB,
                                 lowF1,spnF1,
                                 lowV1,spnV1,
                                 loglowS1,logspnS1)
            
        return logprob, logprob_resample
    
    def make_log_prob_2G_V22xB2x(self):
        # F21 F22 V21 S21 S22
        #   0   1   2   3   4
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y

        BB       = self.BB

        delta_v = self.delta_v
            
        logprob_njit = log_prob_2G_V22xB2x_njit_linear
        lowF21 = self.low[0]; spnF21 = self.spn[0]
        lowF22 = self.low[1]; spnF22 = self.spn[1]
        lowV21 = self.low[2]; spnV21 = self.spn[2]
        lowS21 = self.low[3]; spnS21 = self.spn[3]
        hihS22 = self.hih[4]
        loghihS22 = np.log(hihS22)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF21,uF22,uV21,uS21,uS22 = ps
            return logprob_njit(x,y,inv_e_y,
                                uF21,uF22,uV21,uS21,uS22,BB,
                                lowF21,spnF21,
                                lowF22,spnF22,
                                lowV21,spnV21,
                                lowS21,spnS21,
                                loghihS22,delta_v)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps,y):
            uF21,uF22,uV21,uS21,uS22 = ps
            return -logprob_njit(x,y,inv_e_y,
                                uF21,uF22,uV21,uS21,uS22,BB,
                                lowF21,spnF21,
                                lowF22,spnF22,
                                lowV21,spnV21,
                                lowS21,spnS21,
                                loghihS22,delta_v)
            
        return logprob, logprob_resample
                
    def make_log_prob_2G_V22x(self):
        # F21 F22 V21 S21 S22 B2
        #   0   1   2   3   4  5
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y
        delta_v = self.delta_v
            
        logprob_njit = log_prob_2G_V22x_njit_linear
        lowF21 = self.low[0]; spnF21 = self.spn[0]
        lowF22 = self.low[1]; spnF22 = self.spn[1]
        lowV21 = self.low[2]; spnV21 = self.spn[2]
        lowS21 = self.low[3]; spnS21 = self.spn[3]
        hihS22 = self.hih[4];
        loghihS22 = np.log(hihS22)
        lowB2  = self.low[5]; spnB2  = self.spn[5]
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF21,uF22,uV21,uS21,uS22,uB2 = ps
            return logprob_njit(x,y,inv_e_y,
                                uF21,uF22,uV21,uS21,uS22,uB2,
                                lowF21,spnF21,
                                lowF22,spnF22,
                                lowV21,spnV21,
                                lowS21,spnS21,
                                loghihS22,
                                lowB2,spnB2,delta_v)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps,y):
            uF21,uF22,uV21,uS21,uS22,uB2 = ps
            return -logprob_njit(x,y,inv_e_y,
                                uF21,uF22,uV21,uS21,uS22,uB2,
                                lowF21,spnF21,
                                lowF22,spnF22,
                                lowV21,spnV21,
                                lowS21,spnS21,
                                loghihS22,
                                lowB2,spnB2,delta_v)
            
        return logprob, logprob_resample
                
    
    # 3G ==========================================================================
   
    def make_log_prob_3G_V32xV33xB3x(self):
        # F31 F32 F33 V31 S31 S32 S33
        #   0   1   2   3   4   5   6
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y

        BB       = self.BB
        delta_v  = self.delta_v
            
        logprob_njit = log_prob_3G_V32xV33xB3x_njit_linear
        lowF31=self.low[0]; spnF31=self.spn[0]
        lowF32=self.low[1]; spnF32=self.spn[1]
        lowF33=self.low[2]; spnF33=self.spn[2]
        lowV  =self.low[3]; spnV  =self.spn[3]
        lowS31=self.low[4]
        lowS32=self.low[5]; spnS32=self.spn[5]
        hihS33=self.hih[6]
        loghihS33=np.log(hihS33)
        delta_v = self.delta_v
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF31,uF32,uF33,uV31,uS31,uS32,uS33 = ps
            return logprob_njit(
                x, y, inv_e_y,
                uF31,uF32,uF33,uV31,uS31,uS32,uS33,BB,
                lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                lowV,spnV,
                lowS31,
                lowS32,spnS32,
                loghihS33,delta_v)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps, y):
            uF31,uF32,uF33,uV31,uS31,uS32,uS33 = ps
            return -logprob_njit(
                x, y, inv_e_y,
                uF31,uF32,uF33,uV31,uS31,uS32,uS33,BB,
                lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                lowV,spnV,
                lowS31,
                lowS32,spnS32,
                loghihS33,delta_v)
            
        return logprob, logprob_resample
            
    def make_log_prob_3G_V32xV33x(self):
        # F31 F32 F33 V31 S31 S32 S33 B3
        #   0   1   2   3   4   5   6  7
        
        x        = self.x
        y        = self.y
        inv_e_y  = self.inv_e_y

        delta_v  = self.delta_v
            
        logprob_njit = log_prob_3G_V32xV33x_njit_linear
        lowF31=self.low[0]; spnF31=self.spn[0]
        lowF32=self.low[1]; spnF32=self.spn[1]
        lowF33=self.low[2]; spnF33=self.spn[2]
        lowV  =self.low[3];   spnV=self.spn[3]
        lowS31=self.low[4]
        lowS32=self.low[5]; spnS32 = self.spn[5]
        hihS33=self.hih[6];
        loghihS33=np.log(hihS33)
        lowB3 =self.low[7];  spnB3  = self.spn[7]
        delta_v = self.delta_v
        @njit(fastmath=True, cache=True, inline='always')
        def logprob(ps):
            uF31,uF32,uF33,uV31,uS31,uS32,uS33,uB3 = ps
            return logprob_njit(
                x, y, inv_e_y,
                uF31,uF32,uF33,uV31,uS31,uS32,uS33,uB3,
                lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                lowV,spnV,
                lowS31,
                lowS32,spnS32,
                loghihS33,
                lowB3,spnB3,delta_v)
        @njit(fastmath=True, cache=True, inline='always')
        def logprob_resample(ps, y):
            uF31,uF32,uF33,uV31,uS31,uS32,uS33,uB3 = ps
            return -logprob_njit(
                x, y, inv_e_y,
                uF31,uF32,uF33,uV31,uS31,uS32,uS33,uB3,
                lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                lowV,spnV,
                lowS31,
                lowS32,spnS32,
                loghihS33,
                lowB3,spnB3,delta_v)
                
        return logprob, logprob_resample
   
    
    #==============================================================================

    def array_to_dict_guess(self, params):
        return dict(zip(self.names_param, params))

    def log_prob_guess(self, params):
        # param_dict = self.array_to_dict_guess(params)
        # return -1 * self.log_prob(param_dict)
        lp = self.log_prob(params)
        # if ~np.isfinite(lp): return 1e20
        return -1*lp

    def return_bounds_list(self):
        return [self.dict_bound[key] for key in self.names_param]