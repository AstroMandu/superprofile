"""Data-independent compiled entry point shared by every fit worker.

A module-wide source fingerprint also invalidates caches when imported equations
change (Numba does not otherwise track changes in other source files).
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from numba import njit

from .logps import (log_prob_1G_njit_linear, log_prob_1G_B1x_njit_linear,
                    log_prob_2G_V22xB2x_njit_linear, log_prob_2G_V22x_njit_linear,
                    log_prob_2G_B2x_njit_linear,
                    log_prob_3G_V32xV33xB3x_njit_linear,
                    log_prob_3G_V32xV33x_njit_linear)

PARAMETER_COUNTS = (4, 3, 5, 6, 6, 7, 8)
_EQUATION_REVISION = hashlib.sha256(
    Path(__file__).with_name('logps.py').read_bytes()
    + Path(__file__).with_name('logls.py').read_bytes()).hexdigest()


def _make_shared_kernel(revision):
    # Capture only a source fingerprint, never task data. This makes imported
    # equation edits invalidate the cache without a per-call default argument.
    @njit(cache=True, fastmath=True)
    def log_probability(kind, ps, x, y, inv_error, low, high, span, baseline, delta_v):
        """Same equations/priors as logps, with all task data supplied at runtime."""
        if not revision:
            return -np.inf
        if kind == 0:
            return log_prob_1G_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],
                low[0],span[0],low[1],span[1],math.log(low[2]),
                math.log(high[2])-math.log(low[2]),low[3],span[3])
        if kind == 1:
            return log_prob_1G_B1x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],baseline,
                low[0],span[0],low[1],span[1],math.log(low[2]),
                math.log(high[2])-math.log(low[2]))
        if kind == 2:
            return log_prob_2G_V22xB2x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],ps[4],baseline,
                low[0],span[0],low[1],span[1],low[2],span[2],
                low[3],span[3],high[4],delta_v)
        if kind == 3:
            return log_prob_2G_V22x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],ps[4],ps[5],
                low[0],span[0],low[1],span[1],low[2],span[2],
                low[3],span[3],high[4],low[5],span[5],delta_v)
        if kind == 4:
            return log_prob_2G_B2x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],ps[4],ps[5],baseline,
                low[0],span[0],low[1],span[1],low[2],span[2],
                low[3],span[3],low[4],span[4],high[5],delta_v)
        if kind == 5:
            return log_prob_3G_V32xV33xB3x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],ps[4],ps[5],ps[6],baseline,
                low[0],span[0],low[1],span[1],low[2],span[2],low[3],span[3],
                low[4],low[5],span[5],high[6],delta_v)
        if kind == 6:
            return log_prob_3G_V32xV33x_njit_linear(
                x,y,inv_error, ps[0],ps[1],ps[2],ps[3],ps[4],ps[5],ps[6],ps[7],
                low[0],span[0],low[1],span[1],low[2],span[2],low[3],span[3],
                low[4],low[5],span[5],high[6],low[7],span[7],delta_v)
        return -np.inf
    return log_probability


log_probability = _make_shared_kernel(_EQUATION_REVISION)

def float_array(values):
    """Use one writable, contiguous float64 signature, including pandas inputs."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    return array if array.flags.writeable else array.copy()


class KernelLogProbability:
    """Small Python adapter; only the shared numerical kernel is JIT compiled."""
    def __init__(self, model, kind, objective=False):
        self.kind = kind
        self.objective = objective
        self.x, self.y, self.inv_error = model.x, model.y, model.inv_e_y
        self.low, self.high, self.span = model.low, model.hih, model.spn
        self.baseline = 0.0 if model.BB is None else float(model.BB)
        self.delta_v = float(model.delta_v)

    def __call__(self, params, y=None, baseline=None):
        ps = float_array(params)
        if ps.ndim != 1 or ps.size != PARAMETER_COUNTS[self.kind]:
            raise ValueError('Parameter vector does not match the selected Gaussian model')
        data = self.y if y is None else float_array(y)
        if data.ndim != 1 or data.size != self.x.size:
            raise ValueError('Spectrum length does not match the model velocity axis')
        value = log_probability(
            self.kind, ps, self.x, data, self.inv_error, self.low, self.high,
            self.span, self.baseline if baseline is None else float(baseline), self.delta_v)
        return -value if self.objective else value


def warmup_likelihood_kernel():
    """Populate the shared float64 signature using tiny synthetic arrays only."""
    x = np.linspace(-10., 10., 17)
    y = np.zeros_like(x)
    inv_error = np.ones_like(x)
    low = np.ones(8)
    high = np.full(8, 100.)
    span = high-low
    for kind, count in enumerate(PARAMETER_COUNTS):
        log_probability(kind, np.full(count, .5), x,y,inv_error,
                        low,high,span,0.,1.)
