"""Plotfit: EMCEE-based 1G/2G Gaussian fitter (tidied)

Notes:
- Preserves original behavior but fixes a few bugs and sharp edges.
- Highlights of changes:
  * PEP8-ish formatting, clearer structure, and docstrings.
  * Safer handling of NaNs and timers; replaced fragile comparisons.
  * Fixed `thin==0` no-op in `fill_df_emcee`.
  * Centralized autocorr bookkeeping.
  * Cleaned column creation and DataFrame initialization.
  * Minor variable name typo: use `self.bandwidth` (kept alias to old name for safety).
  * Stable arg indexing in `resample`.
  * Guard file I/O if `path_temp`/`path_plot` is None.

Requires sibling modules:
- .class_Plotfitter_gmodel:Gmodel
- .class_Plotfitter_plotter:Plotter
- .subroutines_Plotfitter helpers
"""
from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Iterable, Literal, Tuple

import emcee
import numpy as np
import pandas as pd
import pylab as plt
from scipy.optimize import minimize
from scipy.stats import kurtosis, percentileofscore, skew
from tqdm import tqdm

from .class_Plotfitter_gmodel import Gmodel
from .class_Plotfitter_plotter import Plotter
from .subroutines_Plotfitter import (
    calc_red_chisq_1G,
    calc_red_chisq_2G,
    do_Ftest,
    gauss,
    gaussian_area,
    get_asymmetry_residuals,
    get_mode,
    sort_outliers,
)


class Plotfit:
    """Driver for 1G + 2G Gaussian fits with MCMC and resampling.

    Parameters
    ----------
    df_stacked : pandas.DataFrame
        Columns expected: 'x', 'y', 'e_y'. Rows with y==0 are filtered out.
    dict_disp : dict
        Dispersion summary per group/bin; used to set bounds for S21/S22.
    name_cube : str, optional
    path_plot : str | Path, optional
    path_temp : str | Path, optional
    plot_autocorr : bool
        If True, writes intermediate autocorr plots.
    longest_length : int | None
        Used only for aligned status messages.
    vdisp_low_intrinsic : float
        Lower intrinsic dispersion scale [same unit as x].
    """

    # -----------------------------
    # Construction and utilities
    # -----------------------------
    def __init__(
        self,
        df_stacked: pd.DataFrame,
        dict_disp: dict,
        name_cube: str | None = None,
        path_plot: str | Path | None = None,
        path_temp: str | Path | None = None,
        plot_autocorr: bool = False,
        longest_length: int | None = None,
        vdisp_low_intrinsic: float = 4.0,
    ) -> None:
        # Filter out exact zeros to avoid spikes in likelihood
        df_stacked = df_stacked.loc[df_stacked["y"] != 0].reset_index(drop=True)
        self.df_stacked = df_stacked

        # Basic arrays
        self.x = np.asarray(self.df_stacked["x"], dtype=np.float64)
        self.y = np.asarray(self.df_stacked["y"], dtype=np.float64)
        self.e_y = np.asarray(self.df_stacked["e_y"], dtype=np.float64)

        self.longest_length = 0 if longest_length is None else int(longest_length)
        self.name_cube = name_cube or "Name"

        # Result containers
        self.df = self._init_main_df()
        self.df_resampled = pd.DataFrame({"Name": [self.name_cube]})
        self.df_params = self._init_params_df()

        # External info
        self.dict_disp = dict_disp
        self.list_disp = np.array([dict_disp[i]["disp"] for i in dict_disp.keys()])

        # Paths
        self.path_plot = Path(path_plot) if path_plot is not None else None
        self.path_temp = Path(path_temp) if path_temp is not None else None
        self.plot_autocorr = bool(plot_autocorr)

        # Model config (free/fix flags; see check_config)
        self.dict_params: dict[str, str] = {
            "A21": "free",  # amplitudes free
            "A22": "free",
            "V21": "fix",   # fix -> V21 = V1
            "V22": "fix",   # fix -> V22 = V21
            "S21": "free",
            "S22": "free",
            "B2": "fix",    # fix -> B2 = B1
        }

        # State flags / bookkeeping
        self.GFIT1_success = False
        self.GFIT2_success = False
        self.resample_success = False
        self._timer_start: float | None = None
        self.stat = "Start"

        # MCMC configuration
        self.testlength: int = 1000
        self.truth_from_resampling: bool = False
        self.slope_tau_1G: float = 50
        self.slope_tau_2G: float = 100
        self.maxiter_1G = 50_000
        self.maxiter_2G = 100_000

        # Runtime objects
        self.gmodel: Gmodel | None = None
        self.sampler: emcee.EnsembleSampler | None = None
        self.resampled: np.ndarray | None = None

        # Posteriors summary statistic
        self.statistics: Literal["mean", "median", "mode"] = "median"
        self.skew_thres = 0.7 if self.statistics == "median" else 0.9

        self.vdisp_low_intrinsic = float(vdisp_low_intrinsic)

        # Derived ranges (filled in check_stacked)
        self.xmin = np.nan
        self.xmax = np.nan
        self.ymin = np.nan
        self.ymax = np.nan
        self.chansep = np.nan
        self.bandwidth = np.nan
        # Backward-compat alias (if any external code inspects it)
        self.bandwidh = None  # set after check_stacked

        # Autocorr tracking arrays
        self._reset_autocorr_buffers(maxiter=self.maxiter_2G)

    # -----------------------------
    # DataFrame initializers
    # -----------------------------
    def _init_main_df(self) -> pd.DataFrame:
        cols = [
            "SNR1",
            "SNR2",
            "N1",
            "N2",
            "F-test",
            "F-crit",
            # 1G params
            "B1",
            "A1",
            "V1",
            "S1",
            # 2G params + errors (asymmetric percentiles)
            "B2",
            "e-_B2",
            "e+_B2",
            "A21",
            "e-_A21",
            "e+_A21",
            "A22",
            "e-_A22",
            "e+_A22",
            "V21",
            "e-_V21",
            "e+_V21",
            "V22",
            "e-_V22",
            "e+_V22",
            "S21",
            "e-_S21",
            "e+_S21",
            "S22",
            "e-_S22",
            "e+_S22",
        ]
        df = pd.DataFrame({"Name": [self.name_cube], **{c: [np.nan] for c in cols}})
        return df

    def _init_params_df(self) -> pd.DataFrame:
        cols = {
            "Reliable": "N",
            "SNR2": np.nan,
            "sn": np.nan,
            "sb": np.nan,
            "An": np.nan,
            "Ab": np.nan,
            "At": np.nan,
            "e_sn": np.nan,
            "e_sb": np.nan,
            "e_An": np.nan,
            "e_Ab": np.nan,
            "e_At": np.nan,
            "sn/sb": np.nan,
            "An/At": np.nan,
            "e_sn/sb": np.nan,
            "e_An/At": np.nan,
        }
        return pd.DataFrame({"Name": [self.name_cube], **{k: [v] for k, v in cols.items()}})

    # -----------------------------
    # I/O helpers
    # -----------------------------
    def writestat(self, message: str) -> None:
        if self.path_temp is None:
            return
        try:
            (self.path_temp).mkdir(parents=True, exist_ok=True)
            with open(self.path_temp / f"stat.{self.name_cube}.txt", "w") as f:
                f.write(f"{self.name_cube:>{self.longest_length}} {message}")
        except Exception:
            pass

    def removestat(self) -> None:
        if self.path_temp is None:
            return
        path_stat = self.path_temp / f"stat.{self.name_cube}.txt"
        if path_stat.exists():
            try:
                os.remove(path_stat)
            except Exception:
                pass

    # -----------------------------
    # Diagnostics / plotting
    # -----------------------------
    def print_diagnose_params(self, gmodel: Gmodel, params: np.ndarray) -> None:
        """Print parameter mapping and bound conformity for debugging."""
        if "A21" in gmodel.names_param:
            df_diag = pd.DataFrame()
            df_diag["Index"] = ["orig", "mapd", "fini"]

            params_dict = gmodel.array_to_dict_guess(params)
            params_mapped = gmodel.array_to_dict_guess(gmodel.map_params(params).flatten())

            df_diag["S1"] = [gmodel.S1, None, None]
            for label in gmodel.names_param:
                in_unit = params_dict[label]
                mapped = params_mapped[label]
                fini = (0 < mapped < 1)
                df_diag[label] = [in_unit, mapped, fini]
            if (df_diag.loc[df_diag["Index"] == "fini"] == False).any(axis=None):
                pprint(gmodel.dict_bound)
                print(df_diag.to_string())

            if "B2" in gmodel.names_param:
                df_ampl = pd.DataFrame({
                    "cond": ["ampl"],
                    "fini": [bool(gmodel.test_2G_ampl(params_dict["A21"], params_dict["A22"], params_dict["B2"]))],
                })
                if (df_ampl["fini"] == False).any():
                    print(df_ampl.to_string())

            df_disp = pd.DataFrame({
                "cond": ["disp1"],
                "S1": [gmodel.S1],
                "S21": [params_dict["S21"]],
                "S22": [params_dict["S22"]],
                "S22-S21": [params_dict["S22"] - params_dict["S21"]],
                "delta_disp": [gmodel.delta_disp],
                "fini": [bool(gmodel.test_2G_disp_order(params_dict["S21"], params_dict["S22"]))],
            })
            if (df_disp["fini"] == False).any():
                print(df_disp.to_string())
        else:
            df_diag = pd.DataFrame()
            df_diag["Index"] = ["orig", "mapd", "fini"]
            params_dict = gmodel.array_to_dict_guess(params)
            params_mapped = gmodel.array_to_dict_guess(gmodel.map_params(params).flatten())
            for label in gmodel.names_param:
                in_unit = params_dict[label]
                mapped = params_mapped[label]
                fini = (0 < mapped < 1)
                df_diag[label] = [in_unit, mapped, fini]
            if (df_diag.loc[df_diag["Index"] == "fini"] == False).any(axis=None):
                pprint(gmodel.dict_bound)
                print(df_diag.to_string())

    def _reset_autocorr_buffers(self, maxiter: int) -> None:
        arrlen = int(maxiter / self.testlength)
        self.autocorr_mean = np.full(arrlen + 1, np.nan)
        self.autocorr_max = np.full(arrlen + 1, np.nan)
        self.chekstep = np.full(arrlen + 1, np.nan)
        self.old_tau = np.inf
        self.autocorr_argmax: int | None = None
        self.savename_autocorr: Path | None = None

    def makeplot_autocorr(self, gmodel: Gmodel, converged: bool) -> None:
        if self.path_plot is None:
            return
        self.savename_autocorr = self.path_plot / f"Plotfit_autocorr_{self.name_cube}.png"

        xs = self.chekstep
        fig_ac, ax_ac = plt.subplots()
        ax_ac.plot(xs, xs / self.slope_tau, "--k", label=r"$N$" + f"={self.slope_tau:.0f}" + r"$\tau$")
        ax_ac.plot(xs, self.autocorr_mean, label="mean")
        if self.autocorr_argmax is not None:
            ax_ac.plot(xs, self.autocorr_max, label=f"max ({gmodel.names_param[self.autocorr_argmax]})")
        ax_ac.set_xlabel("Steps")
        ax_ac.set_ylabel(r"$\hat{\tau}$")
        ax_ac.set_title(self.name_cube)
        ax_ac.legend()
        fig_ac.savefig(self.savename_autocorr, dpi=100)

        if converged:
            ax_ac.axvline(int(2 * np.max(self.old_tau)), color="tab:red")
            if self.savename_autocorr.exists():
                try:
                    os.remove(self.savename_autocorr)
                except Exception:
                    pass
        plt.close(fig_ac)

    def _eta_from_autocorr(self, index: int, maxiter: int) -> str:
        # Start timer on first call
        if self._timer_start is None:
            self._timer_start = time.time()
            return "..."

        y1, y2 = self.autocorr_max[index - 1], self.autocorr_max[index]
        x1, x2 = (index - 1) * self.testlength, index * self.testlength
        slope = (y2 - y1) / self.testlength

        if slope > 1.0 / self.slope_tau:
            x_to_go = maxiter - x2
        else:
            # Where y = x/slope_tau intersects current slope from (x1,y1)
            x_intersect = min((slope * x1 - y1) / (slope - 1.0 / self.slope_tau), maxiter)
            x_to_go = x_intersect - x2

        if x_to_go < 0:
            return "Soon"

        now = time.time()
        dt = now - self._timer_start
        self._timer_start = now
        if dt <= 0:
            return "..."
        eta_sec = x_to_go / (self.testlength / dt)
        if not np.isfinite(eta_sec):
            return "..."
        return str(datetime.timedelta(seconds=int(eta_sec)))

    def check_converged(
        self,
        sampler: emcee.EnsembleSampler,
        gmodel: Gmodel,
        generate_plot: bool = False,
    ) -> bool:
        iteration = sampler.iteration
        # `tol=0` returns the raw estimate; catch failures early
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception:
            tau = np.full(len(gmodel.names_param), np.nan)

        if np.any(~np.isfinite(tau)):
            last = sampler.get_chain(flat=True)[-1, :]
            print(self.name_cube, tau, gmodel.array_to_dict_guess(last))
            self.print_diagnose_params(gmodel, last)

        converged = False
        if iteration > self.maxiter:
            converged = True
        else:
            if np.all(np.isfinite(tau)):
                converged = np.all(tau * self.slope_tau < iteration)
                converged &= np.all(np.abs(self.old_tau - tau) / tau < 0.01)

        self.old_tau = tau
        index = int(iteration / self.testlength)
        self.chekstep[index] = iteration
        self.autocorr_mean[index] = np.nanmean(tau)
        self.autocorr_argmax = int(np.nanargmax(tau)) if np.any(np.isfinite(tau)) else None
        if self.autocorr_argmax is not None:
            self.autocorr_max[index] = tau[self.autocorr_argmax]
        eta = self._eta_from_autocorr(index, self.maxiter)
        self.writestat(f"{self.stat} {iteration} {eta}")
        if generate_plot:
            self.makeplot_autocorr(gmodel, converged)

        if converged:
            self._timer_start = None
        return converged

    # -----------------------------
    # 1-Gaussian fit (init + MCMC)
    # -----------------------------
    def makefit_1G_minimize(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        e_y: np.ndarray,
        guess: Iterable[float] | None = None,
        return_gmodel: bool = False,
    ) -> Tuple[dict, Gmodel] | dict:
        """Quick Nelder-Mead to seed the MCMC for 1G."""
        if self.dict_params["V21"] == 0:
            names_param = np.array(["A1", "S1", "B1"])  # no V1
        else:
            names_param = np.array(["A1", "V1", "S1", "B1"])  # with V1

        dict_bound = {
            "A1": np.array([0, 1.5]) * float(np.nanmax(yy)),
            "V1": np.array([-3 * self.chansep, 3 * self.chansep]),
            "S1": np.array([self.dispmin, self.dispmax]),
            "B1": np.array([-3.0, 3.0]) * float(np.nanmax(yy)),
        }

        gmodel = Gmodel(xx, yy, e_y, names_param, dict_bound)
        gmodel.log_prob = gmodel.log_prob_1G

        if guess is None:
            guess = np.array([np.nanmax(yy), 0.0, 10.0, 0.0])[: len(names_param)]

        res = minimize(gmodel.log_prob_guess, guess, method="Nelder-Mead")
        if return_gmodel:
            return gmodel.array_to_dict_guess(res.x), gmodel
        return gmodel.array_to_dict_guess(res.x)

    def limit_range(self, multiplier_disp: float = 10) -> None:
        """Trim x-range to ~±multiplier_disp × S1 around V1 from a quick 1G fit."""
        res_1G = self.makefit_1G_minimize(self.x, self.y, self.e_y)
        V1, S1 = float(res_1G["V1"]), float(res_1G["S1"])

        xi = max(V1 - S1 * multiplier_disp, float(self.xmin))
        xf = min(V1 + S1 * multiplier_disp, float(self.xmax))

        df_limited = self.df_stacked.loc[self.df_stacked["x"].between(xi, xf)]
        self.x = np.asarray(df_limited["x"], dtype=float)
        self.y = np.asarray(df_limited["y"], dtype=float)
        self.e_y = np.asarray(df_limited["e_y"], dtype=float)

        self.xmin, self.xmax = map(float, np.nanpercentile(self.x, [0, 100]))
        self.ymin, self.ymax = map(float, np.nanpercentile(self.y, [0, 100]))
        self.chansep = float(np.abs(np.mean(np.diff(self.x))))
        self.bandwidth = float(self.xmax - self.xmin)
        self.bandwidh = self.bandwidth  # alias

    def fill_df_emcee(self, sampler: emcee.EnsembleSampler, names: np.ndarray, burnin: int, thin: int = 1) -> None:
        if thin == 0:
            thin = 1
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_probs = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
        max_prob_index = int(np.nanargmax(log_probs))

        if self.statistics == "median":
            for i, label in enumerate(names):
                p16, p50, p84 = np.percentile(flat_samples[:, i], [16, 50, 84])
                self.df[label] = flat_samples[max_prob_index, i]
                self.df[f"e-_{label}"] = p50 - p16
                self.df[f"e+_{label}"] = p84 - p50
        elif self.statistics == "mode":
            for i, label in enumerate(names):
                mode = get_mode(flat_samples[:, i])
                percentile_mode = percentileofscore(flat_samples[:, i], mode)
                percentile_mode = np.clip(percentile_mode, 0.0, 100.0)
                lo, md, hi = np.percentile(
                    flat_samples[:, i],
                    [max(percentile_mode - 34, 0), percentile_mode, min(percentile_mode + 34, 100)],
                )
                self.df[label] = md
                self.df[f"e-_{label}"] = md - lo
                self.df[f"e+_{label}"] = hi - md

        # Fill implicit params when not sampled
        if ("A1" in names) and ("V1" not in names):
            self.df["V1"] = 0.0
        if ("A21" in names) and ("V21" not in names):
            self.df["V21"] = self.df["V1"][0]
        if ("A21" in names) and ("V22" not in names):
            self.df["V22"] = self.df["V21"][0]
        if ("A21" in names) and ("B2" not in names):
            self.df["B2"] = self.df["B1"][0]

        if "A21" in names:
            self.flat_samples = flat_samples

    def makefit_1G(
        self,
        xx: np.ndarray | None = None,
        yy: np.ndarray | None = None,
        e_y: np.ndarray | None = None,
        guess: Iterable[float] | None = None,
    ) -> None:
        self.maxiter = self.maxiter_1G
        self.slope_tau = self.slope_tau_1G

        xx = self.x if xx is None else xx
        yy = self.y if yy is None else yy
        e_y = self.e_y if e_y is None else e_y

        res_1G, gmodel = self.makefit_1G_minimize(xx, yy, e_y, guess=guess, return_gmodel=True)
        guess_vec = [res_1G[param] for param in gmodel.names_param]
        self.gmodel = gmodel

        self.stat = "1GFIT"
        nwalkers = max(4, int(len(guess_vec) * 2))
        ndim = len(guess_vec)

        # Clip initial walkers to bounds
        lb = np.array([gmodel.dict_bound[p][0] for p in gmodel.names_param])
        ub = np.array([gmodel.dict_bound[p][1] for p in gmodel.names_param])
        pos = np.clip(np.array(guess_vec) + 1e-5 * np.random.randn(nwalkers, ndim), lb, ub)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            gmodel.log_prob_1G,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        )

        # Reset autocorr tracking for this run length
        self._reset_autocorr_buffers(self.maxiter)

        for _ in sampler.sample(pos, iterations=self.maxiter):
            if sampler.iteration % self.testlength:
                continue
            if self.check_converged(sampler, gmodel, generate_plot=False):
                break

        # Summarize
        self.burnin = int(2 * np.nanmax(self.old_tau))
        self.thin = max(1, int(0.5 * np.nanmin(self.old_tau)))
        self.fill_df_emcee(sampler, gmodel.names_param, self.burnin, self.thin)
        self.writestat(f"{self.stat} - Done")

        self.df_params["S1"] = self.df["S1"][0]

        A1, V1, S1, B1 = (self.df.loc[0, k] for k in ["A1", "V1", "S1", "B1"])
        residual = yy - (gauss(xx, A1, V1, S1) + B1)
        self.residual_1GFIT = residual

        N1 = float(np.nanstd(residual))
        SNR1 = float(A1) / N1 if N1 > 0 else np.nan
        self.df["SNR1"], self.df["N1"] = SNR1, N1
        self.GFIT1_success = True

    # -----------------------------
    # 2-Gaussian fit (MCMC)
    # -----------------------------
    def make_guess(self, dict_1Gfit: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble initial guess vector for 2G from 1G results."""
        dict_1Gfit = dict_1Gfit or {
            "N1": self.df["N1"][0],
            "B1": self.df["B1"][0],
            "A1": self.df["A1"][0],
            "V1": self.df["V1"][0],
            "S1": self.df["S1"][0],
        }
        guess_map = {
            "A21": dict_1Gfit["A1"] * 0.45,
            "A22": dict_1Gfit["A1"] * 0.45,
            "V21": dict_1Gfit["V1"],
            "V22": dict_1Gfit["V1"],
            "S21": dict_1Gfit["S1"],
            "S22": dict_1Gfit["S1"]*1.5,
            "B2": dict_1Gfit["B1"],
        }
        if guess_map['S21']<self.dispmin:
            guess_map['S21'] = dict_1Gfit['S1']
        # if guess_map['S22']>self.dispmax: 
        # Ensure S21 <= S22 at start
        if guess_map["S21"] > guess_map["S22"]:
            guess_map["S21"] = 0.5 * (dict_1Gfit["S1"] + self.dispmin)

        names_param = np.array([k for k, status in self.dict_params.items() if status == "free"])
        guess = np.array([guess_map[p] for p in names_param], dtype=float)
        return guess, names_param
        



    def makefit_2G(self, maxiter: int = 100_000) -> None:
        
        def _idx(names, key):
            idx = np.where(names == key)[0]
            return int(idx[0]) if idx.size > 0 else None

        def enforce_narrow_first(trial, gmodel, margin_frac=1e-9):

            
            """
            Ensure S21 < S22 by swapping the *entire* (A,V,S) triplets if needed.
            Also enforce a tiny strict inequality margin.
            """
            names = gmodel.names_param
            iA1, iV1, iS1 = _idx(names, "A21"), _idx(names, "V21"), _idx(names, "S21")
            iA2, iV2, iS2 = _idx(names, "A22"), _idx(names, "V22"), _idx(names, "S22")

            # If any are missing, nothing to do
            if None in (iA1, iV1, iS1, iA2, iV2, iS2):
                return trial

            s1, s2 = trial[iS1], trial[iS2]
            if not np.isfinite(s1) or not np.isfinite(s2):
                return trial

            if s1 >= s2:
                # swap the entire triplets
                trial[iA1], trial[iA2] = trial[iA2], trial[iA1]
                trial[iV1], trial[iV2] = trial[iV2], trial[iV1]
                trial[iS1], trial[iS2] = s2, s1

            # enforce strict inequality with a tiny margin
            s1, s2 = trial[iS1], trial[iS2]
            if s1 >= s2:
                eps = margin_frac * max(1.0, abs(s2))
                trial[iS1] = min(s2 - eps, s2 * (1.0 - margin_frac))

            # final clip to bounds on S's only (keeps inside box)
            for iS, key in [(iS1, "S21"), (iS2, "S22")]:
                lo, hi = gmodel.dict_bound[key]
                w = hi - lo
                eps = margin_frac * (w if w > 0 else 1.0)
                trial[iS] = np.clip(trial[iS], lo + eps, hi - eps)

            return trial
        
        def clip_to_bounds(pos, gmodel, margin_frac=1e-9):
            """Clip positions to be strictly inside bounds with a tiny margin."""
            names = gmodel.names_param
            lb = np.array([gmodel.dict_bound[p][0] for p in names], dtype=float)
            ub = np.array([gmodel.dict_bound[p][1] for p in names], dtype=float)

            width = ub - lb
            eps = margin_frac * np.where(width > 0, width, 1.0)
            lo = lb + eps
            hi = ub - eps

            return np.minimum(np.maximum(pos, lo), hi)
        
        def sample_initial_from_prior(gmodel, nwalkers, guess=None, max_tries_per=5000, rng=None):
            rng = np.random.default_rng() if rng is None else rng
            names = gmodel.names_param
            ndim  = len(names)
            lb = np.array([gmodel.dict_bound[p][0] for p in names], dtype=float)
            ub = np.array([gmodel.dict_bound[p][1] for p in names], dtype=float)
            
            argwhere_A21 = np.argwhere(names=='A21').item()
            argwhere_A22 = np.argwhere(names=='A22').item()
            argwhere_S21 = np.argwhere(names=='S21').item()
            argwhere_S22 = np.argwhere(names=='S22').item()
            
            
            for i, p in enumerate(names):
                if p=='S22':
                    ub[i] = float(self.df["S1"][0])*3
            
            width = ub - lb
            lb = lb + 0.1 * width
            ub = ub - 0.1 * width

            pos, tries = [], 0
            while len(pos) < nwalkers and tries < max_tries_per * nwalkers:
                tries += 1
                trial = lb + rng.random(ndim) * (ub - lb)
                trial = clip_to_bounds(trial, gmodel)
                trial = enforce_narrow_first(trial, gmodel)          # <— enforce S21<S22 + swap triplets
                
                if np.isin('B2',names):
                    argwhere_B2 = np.argwhere(names=='B2').item()
                    trial_B2 = trial[argwhere_B2]
                else: trial_B2 = self.df.loc[0, 'B2']

                if np.all(np.isfinite(gmodel.log_prior_2G_diagnose(trial))) and gmodel.test_2G_ampl(trial[argwhere_A21],trial[argwhere_A22],trial_B2) and gmodel.test_2G_disp_order(trial[argwhere_S21],trial[argwhere_S22]):
                    pos.append(trial)

            if len(pos) < nwalkers and guess is not None:
                scale = 0.3 * (ub - lb)
                while len(pos) < nwalkers:
                    trial = guess + rng.standard_normal(ndim) * scale
                    trial = clip_to_bounds(trial, gmodel)
                    trial = enforce_narrow_first(trial, gmodel)      # <— also here

                    if np.all(np.isfinite(gmodel.log_prior_2G_diagnose(trial))):
                        pos.append(trial)

            pos = np.asarray(pos)
            pos = clip_to_bounds(pos, gmodel)
            # one last safety pass
            for k in range(pos.shape[0]):
                pos[k] = enforce_narrow_first(pos[k], gmodel)

            return pos



        
        if not self.GFIT1_success:
            self.df_params["Reliable"] = "N"
            return

        self.stat = "2GFIT"
        self.maxiter = self.maxiter_2G if maxiter is None else int(maxiter)
        self.slope_tau = self.slope_tau_2G

        S1 = float(self.df["S1"][0])
        B1 = float(self.df["B1"][0])
        N1 = float(self.df["N1"][0])

        dict_bound = {
            # allow small negatives initially; later priors enforce physicality
            "A21": np.array([-0.001 * self.ymax, self.ymax * 1.5], dtype=float),
            "A22": np.array([-0.001 * self.ymax, self.ymax * 1.5], dtype=float),
            "V21": np.array([-5 * S1, 5 * S1], dtype=float),
            "V22": np.array([-5 * S1, 5 * S1], dtype=float),
            # "S21": np.array([self.dispmin, self.dispmax], dtype=float),
            # "S22": np.array([self.dispmin, self.dispmax], dtype=float),
            "S21": np.array([self.dispmin, S1], dtype=float),
            "S22": np.array([S1, self.dispmax], dtype=float),
            "B2": np.array([B1 - N1, B1 + N1], dtype=float),
        }

        if (dict_bound["S21"][1] - dict_bound["S21"][0]) < 0:
            print(f"[Plotfit] {self.name_cube} 2GFIT no-go; 1GFIT near bound")
            self.df_params["Reliable"] = "nearbound"
            return

        # guess, names_param = self.make_guess()
        # gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)
        # gmodel.log_prob = gmodel.log_prob_2G

        # self.guess_2gfit = guess
        # self.gmodel = gmodel
        # self.params_2gfit = names_param
        # self.dict_bound = dict_bound.copy()

        # # Initial prior sanity
        # if not np.all(np.isfinite(gmodel.log_prior_2G_diagnose(guess))):
        #     pprint(dict_bound)
        #     self.print_diagnose_params(gmodel, guess)
        #     self.make_atlas()
        #     raise ValueError(f"[Plotfit] {self.name_cube}'s initial guess violates priors")

        # nwalkers = max(4, int(len(guess) * 2))
        # ndim = len(guess)
        # lb = np.array([gmodel.dict_bound[p][0] for p in gmodel.names_param])
        # ub = np.array([gmodel.dict_bound[p][1] for p in gmodel.names_param])
        # pos = np.clip(guess + 1e-5 * np.random.randn(nwalkers, ndim), lb, ub)

        # sampler = emcee.EnsembleSampler(
        #     nwalkers,
        #     ndim,
        #     gmodel.log_prob,
        #     moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        # )
        # self.sampler = sampler

        guess, names_param = self.make_guess()
        gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)
        gmodel.log_prob = gmodel.log_prob_2G
        self.guess_2gfit = guess
        self.gmodel = gmodel
        self.params_2gfit = names_param
        self.dict_bound = dict_bound.copy()

        # Check guess validity
        if not np.all(np.isfinite(gmodel.log_prior_2G_diagnose(guess))):
            raise ValueError("initial guess violates priors")

        ndim = len(guess)
        nwalkers = 4 * ndim

        # NEW: initialize from prior
        pos = sample_initial_from_prior(gmodel, nwalkers, guess=guess)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            gmodel.log_prob,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        )
        self.sampler = sampler        
        

        self._reset_autocorr_buffers(self.maxiter)
        for _ in sampler.sample(pos, iterations=self.maxiter):
            if sampler.iteration % self.testlength:
                continue
            if self.check_converged(sampler, gmodel, generate_plot=self.plot_autocorr):
                break

        if self.plot_autocorr and self.savename_autocorr is not None and self.savename_autocorr.exists():
            try:
                os.remove(self.savename_autocorr)
            except Exception:
                pass

        # Summarize
        self.burnin = int(2 * np.nanmax(self.old_tau))
        self.thin = max(1, int(0.5 * np.nanmin(self.old_tau)))
        self.fill_df_emcee(sampler, gmodel.names_param, self.burnin, self.thin)

        # Residuals
        A21, A22, V21, V22, S21, S22, B2 = (self.df.loc[0, k] for k in ["A21", "A22", "V21", "V22", "S21", "S22", "B2"])
        model_narw = gauss(self.x, A21, V21, S21) + B2 / 2.0
        model_brod = gauss(self.x, A22, V22, S22) + B2 / 2.0
        residual = self.y - (model_narw + model_brod)
        self.residual_2GFIT = residual

        self.df["N2"] = float(np.nanstd(residual))
        self.df["SNR2"] = (float(A21) + float(A22)) / self.df["N2"][0] if self.df["N2"][0] > 0 else np.nan

        F, crit = do_Ftest(gmodel, self.df)
        self.df["F-test"], self.df["F-crit"] = F, crit

        # Area metrics
        An, Ab = gaussian_area(A21, S21), gaussian_area(A22, S22)
        self.df_params["SNR2"] = self.df["SNR2"][0]
        self.df_params["sn"], self.df_params["sb"] = S21, S22
        self.df_params["An"], self.df_params["Ab"] = An, Ab
        self.df_params["At"] = An + Ab
        self.df_params["sn/sb"] = self.df_params["sn"][0] / self.df_params["sb"][0]
        self.df_params["An/At"] = self.df_params["An"][0] / self.df_params["At"][0]
        self.df_params["Asym"] = get_asymmetry_residuals(self.y, residual)

        self.GFIT2_success = True
        self.writestat(f"{self.stat} - Done")

    # -----------------------------
    # Resampling
    # -----------------------------
    def resample(self, nsample: int = 1499, pbar_resample: bool = False) -> None:
        SNR1, SNR2 = self.df.loc[0, ["SNR1", "SNR2"]]
        if not (np.isfinite(SNR1) and np.isfinite(SNR2)):
            return

        self.nsample = int(nsample)
        self.df_params["SNR2"] = SNR2
        self.stat = "Resample"

        names_param = np.array(self.params_2gfit)
        guess = np.array([self.df[label][0] for label in names_param], dtype=float)
        resampled = np.full((self.nsample, len(guess)), np.nan, dtype=float)
        S1s = np.full(self.nsample, np.nan, dtype=float)

        guess_1G = self.df.loc[0, ["A1", "V1", "S1", "B1"]].to_numpy(dtype=float)
        N1 = float(self.df["N1"][0])

        pbar = tqdm(total=nsample) if pbar_resample else None
        timei = time.time()

        for j in range(nsample):
            while True:
                # Bootstrap residuals
                y_w_noise = self.y + np.random.choice(self.residual_2GFIT, len(self.y), replace=True)
                ymax = float(np.nanmax(y_w_noise))

                gmodel = self.gmodel  # type: ignore[assignment]
                gmodel.y = y_w_noise

                res_1G = self.makefit_1G_minimize(self.x, y_w_noise, self.e_y, guess=guess_1G)
                A1, V1, S1, B1 = (res_1G[k] for k in ["A1", "V1", "S1", "B1"])

                # Update bounds that depend on data scale
                gmodel.update_bound("A21", np.array([-0.001, 1.5], dtype=np.float64) * ymax)
                gmodel.update_bound("A22", np.array([-0.001, 1.5], dtype=np.float64) * ymax)
                # If B2 were free, could also update bounds using N1
                # gmodel.update_bound("B2", np.array([B1 - N1, B1 + N1], dtype=np.float64))

                res = minimize(gmodel.log_prob_guess, guess, method="Nelder-Mead", tol=1e-8)
                if np.isfinite(res.fun):
                    S1s[j] = float(S1)
                    resampled[j, :] = res.x
                    break

            if j % 100 == 0 and j != 0:
                timef = time.time()
                eta = datetime.timedelta(seconds=int((nsample - j) / (100 / (timef - timei))))
                self.writestat(f"Resample {j} {eta}")
                self.resampled = resampled[:j, :]
                timei = timef

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

        self.resampled = resampled

        # Fill resampled summary
        for i, label in enumerate(names_param):
            data_label = resampled[:, i]
            self.df_resampled[label] = np.nanmean(data_label)
            self.df_resampled[f"e_{label}"] = np.nanstd(data_label)

        # Stable index helpers
        idx = {name: int(np.where(self.params_2gfit == name)[0][0]) for name in ["S21", "S22", "A21", "A22"]}
        sns = resampled[:, idx["S21"]]
        sbs = resampled[:, idx["S22"]]
        Ans = gaussian_area(resampled[:, idx["A21"]], sns)
        Abs = gaussian_area(resampled[:, idx["A22"]], sbs)
        Ats = Ans + Abs

        if self.truth_from_resampling:
            print(f"[Plotfit {self.name_cube}] truth_from_resampling=True; filling df with resampled values")
            for param in self.gmodel.names_param:  # type: ignore[union-attr]
                p_idx = int(np.where(self.params_2gfit == param)[0][0])
                self.df[param] = np.median(sort_outliers(resampled[:, p_idx]))

        # Update df_params with central values
        A21, A22, S21, S22 = (self.df.loc[0, k] for k in ["A21", "A22", "S21", "S22"])
        An, Ab = gaussian_area(A21, S21), gaussian_area(A22, S22)
        self.df_params["sn"], self.df_params["sb"] = S21, S22
        self.df_params["An"], self.df_params["Ab"] = An, Ab
        self.df_params["At"] = An + Ab
        
        self.df_params["e_S1"] = np.nanstd(sort_outliers(S1s))
        self.df_params["e_sn"] = np.nanstd(sort_outliers(sns))
        self.df_params["e_sb"] = np.nanstd(sort_outliers(sbs))
        self.df_params["e_An"] = np.nanstd(sort_outliers(Ans))
        self.df_params["e_Ab"] = np.nanstd(sort_outliers(Abs))
        self.df_params["e_At"] = np.nanstd(sort_outliers(Ats))

        if self.truth_from_resampling:
            # Aggressive outlier handling used previously; keep behavior
            self.df_params["sn/sb"] = np.median(sort_outliers(sns/sbs))
            self.df_params["An/At"] = np.median(sort_outliers(Ans/Ats))
            self.df_params["log(sb-sn)"] = np.nanmedian(sort_outliers(np.log10(sbs - sns)))
        else:
            self.df_params["sn/sb"] = self.df_params["sn"][0] / self.df_params["sb"][0]
            self.df_params["An/At"] = self.df_params["An"][0] / self.df_params["At"][0]
            self.df_params["log(sb-sn)"] = np.log10(self.df_params["sb"][0] - self.df_params["sn"][0])

        self.df_params["e_sn/sb"] = np.nanstd(sort_outliers(sns/sbs))
        self.df_params["e_An/At"] = np.nanstd(sort_outliers(Ans/Ats))
        self.df_params["e_log(sb-sn)"] = np.nanstd(sort_outliers(np.log10(sbs - sns)))

        self.resample_success = True
        self.df_params["Reliable"] = "Y"

    # -----------------------------
    # Checks / guards
    # -----------------------------
    def check_config(self) -> None:
        # Enforce allowed values
        self.dict_params["A21"] = "free"
        self.dict_params["A22"] = "free"
        self.dict_params["S21"] = "free"
        self.dict_params["S22"] = "free"

        if self.dict_params["V21"] == 0:
            self.dict_params["V21"] = "0"

        if self.dict_params["V21"] not in ["0", "free", "fix"]:
            raise TypeError('[Plotfit] V21 must be one of {"0","free","fix"}.')
        if self.dict_params["V22"] not in ["0", "free", "fix"]:
            raise TypeError('[Plotfit] V22 must be one of {"0","free","fix"}.')
        if self.dict_params["B2"] not in ["free", "fix"]:
            raise TypeError('[Plotfit] B2 must be one of {"free","fix"}.')

        if self.dict_params["V21"] == "0" and self.dict_params["V22"] == "free":
            print("[Plotfit] Setting V22=fix; V22 free not possible when V21==0")
            self.dict_params["V22"] = "fix"
        if self.dict_params["V21"] == "fix" and self.dict_params["V22"] == "free":
            print("[Plotfit] Setting V22=fix; V22 free not possible when V21==fix")
            self.dict_params["V22"] = "fix"

    def check_stacked(self) -> bool:
        if np.all(self.y == 0) or np.all(~np.isfinite(self.y)):
            self.df_params["Reliable"] = "0stacked"
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Nothing seems to be stacked.")
            return False

        self.xmin, self.xmax = map(float, np.nanpercentile(self.x, [0, 100]))
        self.ymin, self.ymax = map(float, np.nanpercentile(self.y, [0, 100]))
        self.chansep = float(np.abs(np.mean(np.diff(self.x))))
        self.bandwidth = float(self.xmax - self.xmin)
        self.bandwidh = self.bandwidth  # alias for any external ref
        return True

    def check_list_disp(self) -> bool:
        # Bounds from spectral resolution & bandwidth
        self.dispmin = (self.chansep / 2.355) * 3.0
        self.dispmax = self.bandwidth / 2.355

        if (self.dispmax - self.dispmin) < 2:
            self.df_params["Reliable"] = "disprng"
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Range of stacked dispersion is narrow")
            return False
        return True

    # -----------------------------
    # Post-fit evaluations
    # -----------------------------
    def evaluate_1GFIT(self) -> None:
        A1, V1, S1, B1, SNR1 = self.df.loc[0, ["A1", "V1", "S1", "B1", "SNR1"]]
        # Guard: bandwidth must cover ~FWHM*3
        if S1 * 2.3 * 2 > self.bandwidth:
            print(f"[Plotfit] {self.name_cube} 2GFIT no-go; Bandwidth too narrow")
            self.df_params["Reliable"] = "W_S1>BW"
            self.GFIT1_success = False
            return

    def evaluate_2GFIT(self) -> None:
        A21, A22, N2, SNR2 = self.df.loc[0, ["A21", "A22", "N2", "SNR2"]]
        if float(SNR2) < 15:
            self.df_params["Reliable"] = "lowSNR2"
            self.GFIT2_success = False
            return
        if float(A21) < 3 * float(N2):
            self.df_params["Reliable"] = "lowA21"
            self.GFIT2_success = False
            return
        if float(A22) < 3 * float(N2):
            self.df_params["Reliable"] = "lowA22"
            self.GFIT2_success = False
            return

        # Skewness/kurtosis diagnostics on posteriors (after outlier trimming)
        for param in ["A21", "A22", "S21", "S22"]:
            p_idx = int(np.where(self.params_2gfit == param)[0][0])
            params = sort_outliers(self.flat_samples[:, p_idx])  # type: ignore[index]
            self.df[f"Skew_{param}"] = skew(params)
            self.df[f"Kurt_{param}"] = kurtosis(params)

        skew_S21 = self.df["Skew_S21"][0]
        skew_S22 = self.df["Skew_S22"][0]

        if (skew_S21 > self.skew_thres) and not (skew_S22 < -self.skew_thres):
            self.df_params["Reliable"] = "S21_at_edge"
            self.GFIT2_success = False
            return
        if not (skew_S21 > self.skew_thres) and (skew_S22 < -self.skew_thres):
            self.df_params["Reliable"] = "S22_at_edge"
            self.GFIT2_success = False
            return
        if (skew_S21 < -self.skew_thres) and (skew_S22 > self.skew_thres):
            self.df_params["Reliable"] = "S21~S22"
            self.GFIT2_success = False
            return
        if (abs(skew_S21) > self.skew_thres) and (abs(skew_S22) > self.skew_thres):
            self.df_params["Reliable"] = "disps_at_edge"
            self.GFIT2_success = False
            return

    # -----------------------------
    # Plotter wrapper
    # -----------------------------
    def make_atlas(self) -> None:
        if self.path_plot is None:
            return
        plotter = Plotter(
            self.path_plot,
            self.name_cube,
            self.suffix,
            self.list_disp,
            self.df,
            self.df_params,
            self.gmodel,
            self.sampler,
            self.resampled,
        )
        plotter.makeplot_atlas()

    # -----------------------------
    # Public run method
    # -----------------------------
    def run(self, suffix: str = "", nsample_resample: int = 1499, pbar_resample: bool = False) -> None:
        self.suffix = suffix
        self.nsample_resample = int(nsample_resample)

        self.check_config()
        if not self.check_stacked():
            return
        if not self.check_list_disp():
            return

        self.limit_range(multiplier_disp=15)
        self.makefit_1G(self.x, self.y, self.e_y)
        self.evaluate_1GFIT()
        self.make_atlas()

        if self.GFIT1_success:
            self.makefit_2G()
            self.make_atlas()
            self.resample(nsample=nsample_resample, pbar_resample=pbar_resample)
            self.evaluate_2GFIT()
            self.make_atlas()

        self.removestat()
