import numpy as np
from .subroutines_Plotfitter import model_1G, model_2G, linmap, chisq_gauss2, test_off_bounds, bound_to_div, log_L_1G_jit, log_L_2G_jit, _softplus, _sigmoid_mapped, _softabs, _clip_raw, _idx,check_sanity, check_sanity_softplus
from math import isfinite
from pprint import pprint

S_EPS  = np.float64(1e-12) 
S_MAX = np.float64(1e3)
RAW_LIM = np.float64(40.0)

class Gmodel:

    def __init__(self, 
                 xx: np.ndarray, yy: np.ndarray, e_y: np.ndarray,
                 names_param, dict_bound,
                 df_plotfit=None):

        # Ensure consistent float64 typing for Numba compatibility and performance
        self.x = np.asarray(xx, dtype=np.float64)
        self.y = np.asarray(yy, dtype=np.float64)
        self.e_y = np.asarray(e_y, dtype=np.float64)
        self.inv_e_y = 1. / self.e_y

        self.delta_disp = 0.
        
        self._low  = np.array([dict_bound[k][0]            for k in names_param])
        self._div  = np.array([bound_to_div(dict_bound[k]) for k in names_param])

        if df_plotfit is not None:
            self.S1 = np.float64(df_plotfit.loc[0, 'S1'])
            self.B1 = np.float64(df_plotfit.loc[0, 'B1'])
            self.V1 = np.float64(df_plotfit.loc[0, 'V1'])
            
        if np.isin('A1',names_param):
            self.iA1 = _idx(names_param,'A1')
            self.iS1 = _idx(names_param,'S1')
            self.iB1 = _idx(names_param,'B1')
            self.has_V1 = False
            if np.isin('V1',names_param):
                self.iV1 = _idx(names_param,'V1')
                self.has_V1 = True
        if np.isin('A21',names_param):
            self.iA21 = _idx(names_param,'A21')
            self.iA22 = _idx(names_param,'A22')
            self.iS21 = _idx(names_param,'S21')
            self.iS22 = _idx(names_param,'S22')
            self.has_V21 = False
            self.has_V22 = False
            self.has_B2  = False
            if np.isin('V21',names_param):
                self.iV21 = _idx(names_param,'V21')
                self.has_V21 = True
            if np.isin('V22',names_param):
                self.iV22 = _idx(names_param,'V22')
                self.has_V22 = True
            if np.isin('B2',names_param):
                self.iB2 = _idx(names_param,'B2')
                self.has_B2 = True

        self.df = df_plotfit
        self.names_param = names_param
        self.dict_bound = dict_bound

    # @profile
    # def map_params(self, params):
    #     arr = np.empty(len(self.names_param), dtype=np.float64)
    #     for i, key in enumerate(self.names_param):
    #         val = params[key]
    #         low = self.dict_bound[key][0]
    #         div = self.dict_bound['div' + key]
    #         arr[i] = linmap(val, low, div)
    #     return arr
    
    def update_bound(self, name_param, bound):
        argwhere = np.argwhere(self.names_param==name_param).item()
        self._low[argwhere] = bound[0]
        self._div[argwhere] = 1.0 / (bound[1] - bound[0])
        self.dict_bound[name_param] = bound
    
    # def map_params(self, params):
    #     return linmap(params, self._low, self._div)
    
    def map_params(self, params):
        return (params - self._low) * self._div

    def test_2G_ampl(self,A21,A22,B2):
        if A21 + A22 + B2 > self.dict_bound['A21'][1]: 
            return False
        return True
    
    def test_2G_disp_order(self,S21,S22):
        if S22 - S21 < self.delta_disp:
            return False
        return True

    def log_L_1G(self, A1, V1, S1, B1):
        return log_L_1G_jit(self.x, self.y, self.inv_e_y, A1, V1, S1, B1)

    def log_L_2G(self, A21, A22, V21, V22, S21, S22, B2):
        return log_L_2G_jit(self.x, self.y, self.inv_e_y, A21, A22, V21, V22, S21, S22, B2)

    def log_prob_1G(self, params):
        A1,S1,B1 = params[self.iA1], params[self.iS1], params[self.iB1]
        if self.has_V1: V1=params[self.iV1] 
        else: V1=self.V1
        mapped = self.map_params(params)
        if test_off_bounds(mapped): return -np.inf
        return self.log_L_1G(A1,V1,S1,B1)
    
    def log_prob_1G_unconstrained(self, params):
        if not check_sanity(params): return -np.inf
        A1,S1,B1 = params[self.iA1], params[self.iS1], params[self.iB1]
        if self.has_V1: V1=params[self.iV1] 
        else: V1=self.V1
        dict_bound = self.dict_bound
        uA1 = _sigmoid_mapped(A1,dict_bound['A1'])
        uV1 = _sigmoid_mapped(V1,dict_bound['V1'])
        uS1 = _sigmoid_mapped(S1,dict_bound['S1'])
        uB1 = _sigmoid_mapped(B1,dict_bound['B1'])
        logl = self.log_L_1G(uA1,uV1,uS1,uB1)
        return logl if isfinite(logl) else -np.inf
    
    def log_prob_2G(self, params):
        A21,A22,S21,S22 = params[self.iA21],params[self.iA22],params[self.iS21],params[self.iS22]
        if self.has_V21: V21=params[self.iV21] 
        else: V21=self.V1
        if self.has_V22: V22=params[self.iV22] 
        else: V22=V21
        if self.has_B2:  B2=params[self.iB2]   
        else: B2=self.B1
        mapped = self.map_params(params)
        if test_off_bounds(mapped): return -np.inf
        if self.test_2G_disp_order(S21,S22)==False: return -np.inf
        if self.test_2G_ampl(A21,A22,B2)==False:    return -np.inf
        return self.log_L_2G(A21,A22,V21,V22,S21,S22,B2)
    
    def log_prob_2G_unconstrained(self, params):
        if not check_sanity(params): return -np.inf
        A21,A22,S21,S22 = (
            params[self.iA21],
            params[self.iA22],
            params[self.iS21],
            params[self.iS22],
        )
        V21 = params[self.iV21] if self.has_V21 else self.V1
        V22 = params[self.iV22] if self.has_V22 else V21
        B2  = params[self.iB2 ] if self.has_B2  else self.B1
        
        dict_bound = self.dict_bound
        boundA21 = dict_bound['A21']
        boundA22 = dict_bound['A22']
        boundV2X = dict_bound['V21']
        boundS2X = dict_bound['S21']
        boundB2  = dict_bound['B2']
        
        uA21 = _sigmoid_mapped(A21,boundA21)
        uA22 = _sigmoid_mapped(A22,boundA22)
        uV21 = _sigmoid_mapped(V21,boundV2X)
        uV22 = _sigmoid_mapped(V22,boundV2X)
        uS21 = _sigmoid_mapped(S21,boundS2X)
        uS22 = _sigmoid_mapped(S22,boundS2X)
        uB2  = _sigmoid_mapped(B2, boundB2)
        
        logl = self.log_L_2G(uA21,uA22,uV21,uV22,uS21,uS22,uB2)
        return logl if isfinite(logl) else -np.inf

    def array_to_dict_guess(self, params):
        return dict(zip(self.names_param, params))
    
    def log_prior_2G_diagnose(self, guess):
        mapped = self.map_params(guess)
        if test_off_bounds(mapped): return -np.inf
        return 0.0

    def log_prob_guess(self, params):
        # param_dict = self.array_to_dict_guess(params)
        # return -1 * self.log_prob(param_dict)
        lp = self.log_prob(params)
        if ~np.isfinite(lp): return 1e20
        return -lp

    def return_bounds_list(self):
        return [self.dict_bound[key] for key in self.names_param]