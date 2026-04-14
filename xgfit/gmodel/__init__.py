from .gmodel  import Gmodel
from .mappers import map_params
from .logl_njit import (log_prob_1G_njit_linear,
                        log_prob_1G_B1x_njit_linear,
                        log_prob_2G_V22x_njit_linear,
                        log_prob_2G_V22xB2x_njit_linear,
                        log_prob_3G_V32xV33x_njit_linear,
                        log_prob_3G_V32xV33xB3x_njit_linear)