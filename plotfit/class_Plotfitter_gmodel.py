import numpy as np
from .subroutines_Plotfitter import model_1G, model_2G, linmap, chisq_gauss2, test_off_bounds, bound_to_div, log_L_1G_jit, log_L_2G_jit
from pprint import pprint

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
            self.argwhere_A1 = np.argwhere(names_param=='A1').item()
            self.argwhere_S1 = np.argwhere(names_param=='S1').item()
            self.argwhere_B1 = np.argwhere(names_param=='B1').item()
            self.has_V1 = False
            if np.isin('V1',names_param):
                self.argwhere_V1 = np.argwhere(names_param=='V1').item()
                self.has_V1 = True
        if np.isin('A21',names_param):
            self.argwhere_A21 = np.argwhere(names_param=='A21').item()
            self.argwhere_A22 = np.argwhere(names_param=='A22').item()
            self.argwhere_S21 = np.argwhere(names_param=='S21').item()
            self.argwhere_S22 = np.argwhere(names_param=='S22').item()
            self.has_V21 = False
            self.has_V22 = False
            self.has_B2  = False
            if np.isin('V21',names_param):
                self.argwhere_V21 = np.argwhere(names_param=='V21').item()
                self.has_V21 = True
            if np.isin('V22',names_param):
                self.argwhere_V22 = np.argwhere(names_param=='V22').item()
                self.has_V22 = True
            if np.isin('B2',names_param):
                self.argwhere_B2 = np.argwhere(names_param=='B2').item()
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
        A1,S1,B1 = params[self.argwhere_A1], params[self.argwhere_S1], params[self.argwhere_B1]
        if self.has_V1: V1=params[self.argwhere_V1] 
        else: V1=self.V1
        mapped = self.map_params(params)
        if test_off_bounds(mapped): return -np.inf
        return self.log_L_1G(A1,V1,S1,B1)
    
    def log_prob_2G(self, params):
        A21,A22,S21,S22 = params[self.argwhere_A21],params[self.argwhere_A22],params[self.argwhere_S21],params[self.argwhere_S22]
        if self.has_V21: V21=params[self.argwhere_V21] 
        else: V21=self.V1
        if self.has_V22: V22=params[self.argwhere_V22] 
        else: V22=V21
        if self.has_B2:  B2=params[self.argwhere_B2]   
        else: B2=self.B1
        mapped = self.map_params(params)
        if test_off_bounds(mapped): return -np.inf
        if self.test_2G_disp_order(S21,S22)==False: return -np.inf
        if self.test_2G_ampl(A21,A22,B2)==False:    return -np.inf
        return self.log_L_2G(A21,A22,V21,V22,S21,S22,B2)

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