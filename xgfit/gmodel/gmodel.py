import numpy as np
from .kernels import KernelLogProbability, float_array

from ..subroutines import bound_to_span, idx


class Gmodel:

    def __init__(self, 
                 xx: np.ndarray, yy: np.ndarray, e_y: np.ndarray,
                 names_param, dict_bound,
                 df_plotfit=None):

        self.x       = float_array(xx)
        self.y       = float_array(yy)
        self.e_y     = float_array(e_y)
        self.inv_e_y = float_array(1.0 / self.e_y)
        
        self.delta_v = np.abs(np.mean(np.diff(self.x))) / 2.355
        # self.delta_v = np.float64(0.0)
        
        self.low = np.float64([dict_bound[k][0]             for k in names_param])
        self.hih = np.float64([dict_bound[k][1]             for k in names_param])
        # self.div = np.float64([ bound_to_div(dict_bound[k]) for k in names_param])
        self.spn = np.ascontiguousarray(self.hih - self.low, dtype=np.float64)
        
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
    
    def _make_log_probability(self, kind):
        return (KernelLogProbability(self, kind),
                KernelLogProbability(self, kind, objective=True))

    def make_log_prob_1G(self):
        return self._make_log_probability(0)

    def make_log_prob_1G_B1x(self):
        return self._make_log_probability(1)

    def make_log_prob_2G_V22xB2x(self):
        return self._make_log_probability(2)

    def make_log_prob_2G_V22x(self):
        return self._make_log_probability(3)

    def make_log_prob_2G_B2x(self):
        return self._make_log_probability(4)

    def make_log_prob_3G_V32xV33xB3x(self):
        return self._make_log_probability(5)

    def make_log_prob_3G_V32xV33x(self):
        return self._make_log_probability(6)

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
