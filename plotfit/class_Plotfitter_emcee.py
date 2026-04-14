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

import copy
import ctypes
import datetime
import gc
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Iterable, Literal, Tuple

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, spearmanr
from tqdm import tqdm

from .class_Plotfitter_gmodel import Gmodel
from .class_Plotfitter_plotter import Plotter
from .subroutines_Plotfitter import (check_converged_resampling,
                                     gauss, gaussian_area, idx,
                                     makefit_bg_linefree_njit,
                                     sort_outliers,trim_outlier_mahalanobis, compute_waic, compute_aic_bic_aicc)
from .gmodel_mappers import map_params


class Plotfit:
    """Driver for 1G + 2G Gaussian fits with MCMC and resampling.

    Parameters
    ----------
    df_stacked : pandas.DataFrame
        Columns expected: 'x', 'y', 'e_y'. Rows with y==0 are filtered out.
    dict_stacked : dict
        Dispersion summary per group/bin; used to set bounds for S21/S22.
    name_cube : str, optional
    path_plot : str | Path, optional
    path_temp : str | Path, optional
    plot_autocorr : bool
        If True, writes intermediate autocorr plots.
    vdisp_low_intrinsic : float
        Lower intrinsic dispersion scale [same unit as x].
    """

    # -----------------------------
    # Construction and utilities
    # -----------------------------
    def __init__(
        self,
        df_stacked: pd.DataFrame,
        dict_stacked: dict,
        name_cube: str | None = None,
        path_plot: str | Path | None = None,
        path_temp: str | Path | None = None,
        plot_autocorr: bool = False,
        vdisp_low_intrinsic: float = 0.0,
    ) -> None:
        # Filter out exact zeros to avoid spikes in likelihood
        df_stacked = df_stacked.loc[df_stacked["y"] != 0].reset_index(drop=True)
        self.df_stacked = df_stacked

        # Basic arrays
        self.x   = np.asarray(self.df_stacked["x"],   dtype=np.float64)
        self.y   = np.asarray(self.df_stacked["y"],   dtype=np.float64)
        self.e_y = np.asarray(self.df_stacked["e_y"], dtype=np.float64)

        self.name_cube = name_cube or "Name"

        # External info
        self.dict_stacked = dict_stacked
        self.list_NHI_ = np.array([dict_stacked[i]["NHI" ] for i in dict_stacked.keys()])
        self.list_disp = np.array([dict_stacked[i]["disp"] for i in dict_stacked.keys()])

        # Paths
        self.path_plot = Path(path_plot) if path_plot is not None else None
        self.path_temp = Path(path_temp) if path_temp is not None else None
        self.plot_autocorr = bool(plot_autocorr)
        
        if self.path_plot.exists()==False: os.makedirs(self.path_plot)
        if self.path_temp.exists()==False: os.makedirs(self.path_temp)
        
        self.fit_params_1G = {
            "A1":"free",
            "V1":"free",
            "S1":"free",
            "B1":"free",    # fix -> B2 = B1
        }

        self.fit_params_2G = {
            "A21": "free","A22": "free",
            "V21": "free",   # fix -> V21 = V1
            "V22": "V21",   # fix -> V22 = V21
            "S21": "free",
            "S22": "free",
            "B2" : "free",    # fix -> B2 = B1
        }
        
        self.fit_params_3G = {
            "A31":"free","A32":"free","A33":"free",
            "V31":"free",
            "V32":"V31","V33":"V31",
            "S31":"free","S32":"free","S33":"free",
            "B3" :"free",
        }
        

        # State flags / bookkeeping
        self.good_1GFIT = False
        self.good_NGFIT = False
        self.good_resample = False
        self._timer_start: float | None = None
        self.stat = "Start"

        # MCMC configuration
        self.testlength: int = 1000
        self.next_check = 10000
        self.truth_from_resampling: bool = False

        # Runtime objects
        self.gmodel: Gmodel | None = None
        self.resampled: np.ndarray | None = None

        # Posteriors summary statistic
        self.statistics: Literal["MAP",'MEDIAN'] = "MAP"

        self.vdisp_low_intrinsic = float(vdisp_low_intrinsic)

        # Derived ranges (filled in check_stacked)
        self.xmin = np.nan
        self.xmax = np.nan
        self.ymin = np.nan
        self.ymax = np.nan
        self.chansep = np.nan
        self.bandwidth = np.nan
        # Backward-compat alias (if any external code inspects it)
        self.bandwidth = None  # set after check_stacked
        
        self.NGFIT_from_minimize = False
        self.slope_tau = 50
        self.verbose = True
        
        self.method_minimize = 'Nelder-Mead'
        
        self.dict_preguess: dict[str, float] = {}
        self.dict_prebound: dict[str, tuple[float]] = {} 
        
        self.burnin = 0
        self.thin   = 1
        self.sampler = None
        
        self.multiplier_burnin_maxtau = 3.0
        self.fallback_to_2gfit = False
        
        # Result containers
        self._init_main_df()
        self._init_params_df()
    # -----------------------------
    # DataFrame initializers
    # -----------------------------
    def _init_main_df(self) -> pd.DataFrame:
        base_cols = [
            "Nsample", "Burnin",
            "SNR1", "SNR2", "SNR3",
            "N1", "N2", "N3",
            "B1", "A1", "V1", "S1",
            "B2", "A21", "A22", "V21", "V22", "S21", "S22",
            "B3", "A31", "A32", "A33", "V31", "V32", "V33", "S31", "S32", "S33",
        ]

        err_targets = [
            "B2", "A21", "A22", "V21", "V22", "S21", "S22",
            "B3", "A31", "A32", "A33", "V31", "V32", "V33", "S31", "S32", "S33",
        ]
        err_cols = [f"e-_{c}" for c in err_targets] + [f"e+_{c}" for c in err_targets]

        snr_sub_cols = [f"SNR{G}{g}" for G in (2, 3) for g in range(1, G + 1)]
        cols_all = base_cols + err_cols + snr_sub_cols
        self.df = pd.DataFrame(
            {"Name": [self.name_cube], **{c: [np.nan] for c in cols_all}}
        )
        return
    
    def _init_params_df(self) -> pd.DataFrame:
        cols = {
            "Nsample": np.nan,
            "Reliable": "N",
            "SNR2": np.nan,
            'chansep': np.nan,
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
        self.df_params = pd.DataFrame({"Name": [self.name_cube], **{k: [v] for k, v in cols.items()}})
        self.df_params['chansep'] = self.chansep
        return 
    


    # -----------------------------
    # I/O helpers
    # -----------------------------
    def writestat(self, message: str) -> None:
        if self.path_temp is None:
            return
        try:
            (self.path_temp).mkdir(parents=True, exist_ok=True)
            with open(self.path_temp / f"stat.{self.name_cube}{self.suffix}.txt", "w") as f:
                f.write(f"{self.name_cube} {self.suffix} {message}")
        except Exception:
            pass

    def removestat(self) -> None:
        if self.path_temp is None:
            return
        path_stat = self.path_temp / f"stat.{self.name_cube}{self.suffix}.txt"
        if path_stat.exists():
            try:
                os.remove(path_stat)
            except Exception:
                pass

    # -----------------------------
    # Diagnostics / plotting
    # -----------------------------
    def print_diagnose_params(self, taus, params, gmodel: Gmodel) -> None:
        """Print parameter mapping and bound conformity for debugging."""
        
        df_diag = pd.DataFrame()
        df_diag["Index"] = ["tau", "mapd", "demapd", "finite"]
        
        dict_params_mapped   = gmodel.array_to_dict_guess(params)
        dict_params_demapped = gmodel.array_to_dict_guess(map_params(params,gmodel,mode='u->x'))
        
        # print(f'S1 = {self.df.loc[0,"S1"]}')
        
        # np.array([self.dispmin, self.dispmax])
        
        # S1 = self.df.loc[0,"S1"].item()
        # mapd = _inv_sigmoid_mapped(S1, self.dispmin, self.dispmax-self.dispmin)
        # df_diag['S1'] = [mapd,S1,None]
        
        for i,label in enumerate(gmodel.names_param):
            mapd    =   dict_params_mapped[label]
            demapd  = dict_params_demapped[label]
            bound   = gmodel.dict_bound[label]
            fini    = bound[0]<demapd<bound[1]
            df_diag[label] = [taus[i], mapd, demapd, fini]
                
        # print(f'S31<S32<S33: {dict_params_demapped["S31"]<dict_params_demapped["S32"]<dict_params_demapped["S33"]}')
        # print(f'A31>A33 and A32>A33: {dict_params_mapped["A31"]>dict_params_mapped["A33"] and dict_params_mapped["A32"]>dict_params_mapped["A33"]}')
        
        print(f'{self.name_cube} {self.suffix}\n{df_diag.to_string()}')


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
        self.savename_autocorr = self.path_plot / f"Plotfit_autocorr_{self.name_cube}{self.suffix}.png"

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

    def _eta_from_autocorr(self, iteration: int, current_tau_max: float, maxiter: int) -> str:
        now = time.time()
        
        # Initialization on first successful finite tau
        if self._timer_start is None or not np.isfinite(self.old_tau_max):
            self._timer_start = now
            self._last_iter = iteration
            self.old_tau_max = current_tau_max
            return "..."

        # 1. Calculate how many iterations are "left" based on your slope logic
        # x1, y1 = self._last_iter, self.old_tau_max
        # x2, y2 = iteration, current_tau_max
        delta_x = iteration - self._last_iter
        delta_y = current_tau_max - self.old_tau_max
        
        # Avoid div by zero if interval is somehow 0
        if delta_x <= 0: return "..."
        
        slope = delta_y / delta_x
        
        # Your geometric intersection logic
        if slope > 1.0 / self.slope_tau:
            x_to_go = maxiter - iteration
        else:
            # Intersection of line through (x2, y2) with y = x / slope_tau
            denom = (slope - 1.0 / self.slope_tau)
            if abs(denom) < 1e-9: 
                x_to_go = maxiter - iteration
            else:
                x_intersect = (slope * iteration - current_tau_max) / denom
                x_to_go = min(x_intersect, maxiter) - iteration

        if x_to_go <= 0: return "Soon"

        # 2. Calculate time per iteration based on the interval just completed
        dt = now - self._timer_start
        iters_per_sec = delta_x / dt if dt > 0 else 0
        
        if iters_per_sec <= 0: return "..."

        eta_sec = x_to_go / iters_per_sec
        
        # Update trackers for next call
        self._timer_start = now
        self._last_iter = iteration
        self.old_tau_max = current_tau_max
        
        return str(datetime.timedelta(seconds=int(eta_sec)))

    def check_converged(self, sampler, gmodel, generate_plot=False) -> bool:
        iteration = sampler.iteration
        
        # Perform the expensive check
        try:
            tau = sampler.get_autocorr_time(tol=0, thin=self.thin)
        except Exception:
            tau = np.full(len(gmodel.names_param), np.nan)
            
        if np.any(np.isnan(tau)):
            last_coords = sampler.get_last_sample().coords  # (nwalkers, ndim)
            last        = np.nanmean(last_coords, axis=0)   # or choose one walker: last_coords[0]
            self.print_diagnose_params(tau,last,gmodel)

        # Determine convergence
        converged = False
        max_tau = np.nanmax(tau) if np.any(np.isfinite(tau)) else np.nan
        
        if iteration > self.maxiter:
            converged = True
        elif np.all(np.isfinite(tau)):
            cond1 = np.all(tau * self.slope_tau < iteration)
            rel_change = np.abs(self.old_tau - tau) / tau
            cond2 = np.nanmean(rel_change) < 0.1
            converged = cond1 & cond2
            
            if cond1 and not cond2:
                self.next_check = iteration + 200
            else:
                step = int(max_tau * 10)
                self.next_check = iteration + step
                
        # Update ETA using the new variable-aware function
        eta = self._eta_from_autocorr(iteration, max_tau, self.maxiter)
        self.writestat(f"{self.stat} {iteration} {eta}")

        self.old_tau = tau
        # Update burnin and thinning
        if np.isfinite(max_tau):
            self.burnin = int(self.multiplier_burnin_maxtau * max_tau)
            self.thin = max(1, int(0.5 * np.nanmin(tau)))

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
        if self.fit_params_2G["V21"] == 0:
            names_param = np.array(["F1", "S1", "B1"])  # no V1
        else:
            names_param = np.array(["F1", "V1", "S1", "B1"])  # with V1

        mom0 = np.sum(self.y)*self.chansep
        dict_bound = {
            "F1": [0, mom0*1.5],
            "V1": [-10*self.chansep, self.chansep*10],
            "S1": [self.dispmin, self.dispmax],
            "B1": [-1.0* self.ymax, 1.0* self.ymax]
        }
        for k,v in dict_bound.items(): dict_bound[k] = np.float64(v)
        
        if guess is None:
            guess = np.array([mom0, 0.0, 20.0, 0.0])[: len(names_param)]

        gmodel = Gmodel(xx, yy, e_y, names_param, dict_bound)
        
        lp, lpr = gmodel.make_log_prob_1G()
        gmodel.log_prob = lp
        guess = map_params(guess,gmodel,mode='x->u')

        res = minimize(lpr, guess, args=(yy,), method=self.method_minimize)
        
        resx = res.x
        resx = map_params(resx, gmodel, mode='u->x')
            
        if return_gmodel:
            return gmodel.array_to_dict_guess(resx), gmodel
        return gmodel.array_to_dict_guess(resx)

    def limit_range(self, multiplier_disp: float = 10) -> None:
        
        self.writestat(f'RANGE . .')
        
        res_1G = self.makefit_1G_minimize(self.x, self.y, self.e_y)
        V1, S1 = float(res_1G["V1"]), float(res_1G["S1"])

        xi = max(V1 - S1 * multiplier_disp, float(self.xmin))
        xf = min(V1 + S1 * multiplier_disp, float(self.xmax))
        
        # xi = max(V1 - S1*2.355*2.5, float(self.xmin))
        # xf = min(V1 + S1*2.355*2.5, float(self.xmax))

        df_limited = self.df_stacked.loc[self.df_stacked["x"].between(xi, xf)].reset_index(drop=True)
        self.df_stacked = df_limited

        # keep arrays and metadata in sync
        self.x   = np.ascontiguousarray(  self.df_stacked["x"].to_numpy(dtype=float, copy=False))
        self.y   = np.ascontiguousarray(  self.df_stacked["y"].to_numpy(dtype=float, copy=False))
        self.e_y = np.ascontiguousarray(self.df_stacked["e_y"].to_numpy(dtype=float, copy=False))

        self.xmin, self.xmax = map(float, (np.nanmin(self.x), np.nanmax(self.x)))
        self.ymin, self.ymax = map(float, (np.nanmin(self.y), np.nanmax(self.y)))
        self.chansep = float(np.abs(np.mean(np.diff(self.x)))) if self.x.size > 1 else np.nan
        self.bandwidth = float(self.xmax - self.xmin)
        # self.dispmax = self.bandwidth/2. / 2.355
        # self.dispmax = 999
        # self.dispmax = 5*S1 
        
        # self.dispmax = 1000.


    def symmeterise_x(self) -> None:
        # target a symmetric interval around 0 with positive half-width
        # use the smaller absolute bound so we don't create gaps
        half = float(min(abs(self.xmin), abs(self.xmax)))
        xi, xf = -half, half

        df_limited = self.df_stacked.loc[self.df_stacked["x"].between(xi, xf)].reset_index(drop=True)
        self.df_stacked = df_limited

        # keep arrays and metadata in sync
        self.x   = self.df_stacked["x"].to_numpy(dtype=float, copy=False)
        self.y   = self.df_stacked["y"].to_numpy(dtype=float, copy=False)
        self.e_y = self.df_stacked["e_y"].to_numpy(dtype=float, copy=False)

        if self.x.size == 0:
            # fallback: if empty due to pathological bounds, do nothing (or raise)
            return

        self.xmin, self.xmax = map(float, (np.nanmin(self.x), np.nanmax(self.x)))
        self.ymin, self.ymax = map(float, (np.nanmin(self.y), np.nanmax(self.y)))
        self.chansep = float(np.abs(np.mean(np.diff(self.x)))) if self.x.size > 1 else np.nan
        self.bandwidth = float(self.xmax - self.xmin)
        self.dispmax = 99
        # self.dispmax = self.bandwidth / 2.355
        return
        
    def get_residuals(self, G):
        
        try:
            if G==1: 
                A1,V1,S1,B1 = self.df.loc[0,['A1','V1','S1','B1']]
                model_totl = gauss(self.x,A1,V1,S1) + B1
            else:    
                model_totl = np.sum([gauss(self.x,self.df.loc[0,f'A{G}{g}'],self.df.loc[0,f'V{G}{g}'],self.df.loc[0,f'S{G}{g}']) for g in range(1,G+1)],axis=0)+self.df.loc[0,f'B{G}']
        except ZeroDivisionError:
            print(f"{self.header_printmsg} {[f'S{G}{g}={self.df.loc[0,f'S{G}{g}']}' for g in range(1,G+1)]}")
            raise ZeroDivisionError

        residuals = self.y - model_totl
        
        return residuals

    def fill_df_emcee(self, sampler: emcee.EnsembleSampler, gmodel) -> None:
        
        names = gmodel.names_param
        
        def fill_free_params(statistics):
            burnin = self.burnin
            thin = self.thin
            
            # 1. Get the log-probs first to find the MAP index
            # We do this before thinning to find the absolute best point found
            full_log_probs = sampler.get_log_prob(discard=burnin, flat=True)
            best_index = np.argmax(full_log_probs)
            
            # 2. Now get the thinned chain for percentiles
            flat_deconstr = sampler.get_chain(discard=burnin, flat=True, thin=thin)
            flat_physical = map_params(flat_deconstr, gmodel, mode='u->x')
            
            # # 3. Get the physical MAP point from the full (non-thinned) chain
            # # We need to de-constrain the best point found
            # full_chain = sampler.get_chain(discard=burnin, flat=True)
            # map_point_phys = map_params_u_to_x(full_chain[best_index:best_index+1], gmodel)[0]
            
            for i, label in enumerate(names):
                # Always calculate percentiles for errors
                samples = flat_physical[:,i]
                p16, p50, p84 = np.percentile(samples, [16, 50, 84])
                
                if statistics == "MAP":
                    try:
                        kde = gaussian_kde(samples)
                        x_grid = np.linspace(samples.min(), samples.max(), 1000)
                        kde_values = kde.evaluate(x_grid)
                        marginal_map = x_grid[np.argmax(kde_values)]
                        self.df[label] = marginal_map
                    except np.linalg.LinAlgError:
                        self.df[label] = p50
                    
                    # imaxlp = np.argmax(sampler.get_log_prob(discard=burnin, flat=True, thin=thin))
                    # self.df[label] = samples[imaxlp]
                    
                if statistics == "MEDIAN":
                    self.df[label] = p50
                    
                self.df[f"e-_{label}"] = p50 - p16
                self.df[f"e+_{label}"] = p84 - p50
                
            for i, label in enumerate(names):
                par,Gg = label[0],label[1:]
                if par=='F':
                    FF = self.df.loc[0,label]
                    SS = self.df.loc[0,f'S{Gg}']
                    AA = FF/(SS*2.50662827463)
                    self.df[f'A{Gg}'] = AA
                if par=='A':
                    AA = self.df.loc[0,label]
                    SS = self.df.loc[0,f'S{Gg}']
                    FF = AA*SS*2.50662827463
                    self.df[f'F{Gg}'] = FF
                    
            npars = len(names)
            for ii in range(npars):
                for jj in range(ii+1,npars):
                    xx = flat_deconstr[:, ii]
                    yy = flat_deconstr[:, jj]
                    xx,yy = trim_outlier_mahalanobis(xx,yy)
                    self.df[f'rs_{names[ii]}{names[jj]}'],self.df[f'p_rs_{names[ii]}{names[jj]}'] = spearmanr(xx,yy)
                    # colname = f'dcor_{names[ii]}{names[jj]}'
                    # dcorr = dcor_trimmed(flat_deconstr[:, ii], flat_deconstr[:, jj])
                    # self.df[colname] = dcorr
                    
            # self.df['WAIC'] = compute_waic(full_log_probs)
            
            log_l_max = full_log_probs.max()
            self.df['AIC'],self.df['BIC'],self.df['AICc'] = compute_aic_bic_aicc(log_l_max, npars, len(gmodel.x))
        
            del full_log_probs, flat_deconstr
            gc.collect()       
            return
        
        def fill_fixed_params():
            if 'S1'  in names: 
                G=1
            if 'S21' in names: 
                G=2
                if not np.isin('V21', names): self.df['V21'] = self.df['V1'][0]
                if not np.isin('V22', names): self.df['V22'] = self.df['V21'][0]
                if not np.isin('B2',  names):
                    self.df['B2'] = self.df['BB'][0] if 'BB' in self.df else self.df['B1'][0]
            if 'S31' in names: 
                G=3
                if not np.isin('V31', names): self.df['V31'] = self.df[ 'V1'][0]
                if not np.isin('V32', names): self.df['V32'] = self.df['V31'][0]
                if not np.isin('V33', names): self.df['V33'] = self.df['V31'][0]
                if not np.isin('S32', names): self.df['S32'] = self.df['S1'][0]
                if not np.isin('B3',  names): 
                    self.df['B3'] = self.df['BB'][0] if 'BB' in self.df else self.df['B1'][0]
            return G
                    
        def fill_SNR(G):
            residuals        = self.get_residuals(G=G)
            self.df[f"N{G}"] = float(np.sqrt(np.mean(residuals**2)))
            if G==1:
                A1,N1 = self.df.loc[0,['A1','N1']]
                self.df['SNR1'] = A1/N1
            else:
                As = self.df.loc[0,[f'A{G}{g}' for g in range(1,G+1)]]
                                
                NN = self.df.loc[0,f'N{G}']
                self.df[f'SNR{G}'] = np.sum(As) / NN
                
                for g in range(1,G+1):
                    Ag = self.df.loc[0,f'A{G}{g}']
                    self.df[f'SNR{G}{g}'] = Ag / NN
            return    
                    
        names = gmodel.names_param
        
        self.sampler = sampler
        
        fill_free_params(self.statistics)
        G = fill_fixed_params()
        fill_SNR(G)
        


        return

    # -----------------------------
    # 2-Gaussian fit (MCMC)
    # -----------------------------

    def makefit_bg_linefree(self, multiplier_mask_S1=5, yy=None):
        
        V1, S1, B1, N1 = self.df.loc[0, ['V1', 'S1', 'B1', 'N1']]

        # boolean mask instead of argwhere
        mask = (self.x > V1 - multiplier_mask_S1*S1) & (self.x < V1 + multiplier_mask_S1*S1)
        
        if yy is not None:
            y_bgfit = yy[~mask]
        else:
            y_bgfit = self.y[~mask]
        e_y_bgfit = self.e_y[~mask]
        
        # if y_bgfit.size < 10:
        
        if not (self.xmin<(V1-7*S1) or (V1+7*S1)<self.xmax):
            # nothing to fit; fall back to B1
            print(f'{self.header_printmsg} Prefit bg unsuccessful; not enough data outside 1G fit region. Using B1 as XGFIT baseline.')
            self.fit_params_1G['B1']='B1'
            self.fit_params_2G['B2']='B1'
            self.fit_params_3G['B3']='B1'
            # print(f'{self.header_printmsg} Prefit bg unsuccessful; not enough data outside 1G fit region. Going Bfree.')
            # self.fit_params_1G['B1']='free'
            # self.fit_params_2G['B2']='free'
            # self.fit_params_3G['B3']='free'
            return
        
        self.fit_params_1G['B1']='BB'
        self.fit_params_2G['B2']='BB'
        self.fit_params_3G['B3']='BB'
        
        # weighted mean (solution to argmin sum(((y-b)/e)**2))
        w  = 1.0 / e_y_bgfit**2
        BB = np.sum(w * y_bgfit) / np.sum(w)
        # self.BB = BB
        
        self.BB = np.min([BB,B1])
        self.df['BB'] = self.BB
        return BB
    
    def prep_1GFIT(self):
        mom0 = np.sum(self.y) * self.chansep
        dict_bound = {
            "F1": np.float64([0.0, mom0 * 2]),
            # "A1": [0, self.ymax*1.5],
            "V1": [self.xmin, self.xmax],
            "S1": [self.dispmin, self.dispmax],
            "B1": [-self.ymax, self.ymax],
        }
        for k,v in dict_bound.items(): dict_bound[k] = np.float64(v)

        fitB1  = self.fit_params_1G['B1']
        fixB1  = fitB1 in ('B1', 'BB')
        names_param = ['F1', 'V1', 'S1'] if fixB1 else ['F1', 'V1', 'S1', 'B1']

        gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)

        if fixB1:
            gmodel.BB = self.df.loc[0, 'B1'] if fitB1 == 'B1' else self.BB
            lp, lp_resample = gmodel.make_log_prob_1G_B1x()
            guess = np.float64([mom0, 0.0, np.nanmean(self.list_disp)])
        else:
            lp, lp_resample = gmodel.make_log_prob_1G()
            guess = np.float64([mom0, 0.0, np.nanmean(self.list_disp), 0.0])

        gmodel.log_prob          = lp
        gmodel.log_prob_resample = lp_resample
        self.guess_1G            = guess.copy()
        self.gmodel_1G           = copy.deepcopy(gmodel)

        return gmodel, guess
          
    def prep_2GFIT(self):
        
        self.stat = "FIT2G"
        
        F1,A1,V1,S1,B1,N1 = self.df.loc[0,['F1','A1','V1','S1','B1','N1']]
        
        # Fmax = F1 * 1.5
        dict_bound = {
            "F21": [0, F1*1.5],
            "F22": [0, F1*1.5],
            # 'A21': [0, 1.5*A1],
            # 'A22': [0, 1.5*A1],
            "V21": [-S1,S1],
            "V22": [-S1,S1],
            "S21": [self.dispmin, 1.5*S1],
            "S22": [              self.dispmin, self.dispmax],
            "B2":  [-5*N1,5*N1],
        }
        for k,v in dict_bound.items(): dict_bound[k] = np.float64(v)
        
        preguess = self.dict_preguess
        
        dict_guess = {}
        dict_guess['V21'] = preguess['V21'] if 'V21' in preguess and preguess['V21'] is not None else V1
        dict_guess['V22'] = preguess['V22'] if 'V22' in preguess and preguess['V22'] is not None else V1
        dict_guess['B2']  = preguess['B2']  if 'B2'  in preguess and preguess['B2']  is not None else B1
        
        # # 1
        # dict_guess['F21'],dict_guess['F22'] = np.array([1,0])*F1
        # dict_guess['S21'],dict_guess['S22'] = np.array([1.1,1e10])*S1
        
        # 2
        # dict_guess['F21'],dict_guess['F22'] = np.array([0,1])*F1
        # dict_guess['S21'],dict_guess['S22'] = np.array([self.dispmin+0.1,1.1*S1])
        
        #3
        dict_guess['F21'],dict_guess['F22'] = np.array([0.5,0.5])*F1
        # dict_guess['A21'],dict_guess['A22'] = np.array([0.5,0.5])*A1
        dict_guess['S21'],dict_guess['S22'] = np.array([0.5,1.5])*S1

        # used_guesses = [k for k, v in preguess.items() if v is not None]
        # if len(used_guesses)>0:
        #     print(f"{self.header_printmsg} User-supplied guesses are used for {used_guesses}")
        
        
        fitB2 = self.fit_params_2G['B2']
        fixB2 = fitB2 in ('B1','BB')    
        names_param = ['F21','F22','V21','S21','S22']
        if not fixB2: names_param += ['B2']
            
        gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)
        guess  = np.float64([dict_guess[p] for p in names_param])
        
        if fixB2:
            BB = B1 if fitB2=='B1' else self.BB
            gmodel.BB = np.float64(BB)
            lp,lp_resample = gmodel.make_log_prob_2G_V22xB2x()
        else:
            lp,lp_resample = gmodel.make_log_prob_2G_V22x()
                    
        gmodel.log_prob = lp
        gmodel.log_prob_resample = lp_resample
                
        return gmodel,guess
    
    def prep_3GFIT(self):
        
        self.stat = "FIT3G"
        
        F1,A1,V1,S1,B1,N1 = self.df.loc[0,['F1','A1','V1','S1','B1','N1']]
        # self.dispmax = 5*S1
        
        # Fmax = F1 * 1.5
        dict_bound = {
            # 'A31': [-N1,1.5*A1],
            # 'A32': [  0,1.5*A1],
            # 'A33': [-N1,1.5*A1],

            # "F31": [-0.1*F1, Fmax],
            # "F32": [      0, Fmax],
            # "F33": [-0.1*F1, Fmax],
            
            "F31": [-F1*0.1, F1*1.5],
            "F32": [0.1*F1, F1*1.5],
            "F33": [0, F1*1.5],
            
            # 'A31': [0, 1.5*A1],
            # 'A32': [0, 1.5*A1],
            # 'A33': [0, 1.5*A1],
            
            "V31": [-S1,S1],
            "V32": [-S1,S1],
            "V33": [-S1,S1],
            
            # 'S31': [self.dispmin, self.dispmax],
            # 'S32': [self.dispmin, self.dispmax], #[0.4*S1, 1.6*S1],
            # # 'S32': np.float64([0.5*S1,1.5*S1]),
            # # 'S32': np.float64([0.8*S1, self.dispmax]),
            # # 'S32': np.float64([self.dispmin, 1.35*S1]),
            # # 'S32': np.float64([0.8*S1, 1.2*S1]),
            # 'S33': [self.dispmin, self.dispmax],
            
            'S31': [self.dispmin, S1],
            # 'S32': [0.5*S1,   1.5*S1],
            'S32': [1e-5, self.dispmax],
            'S33': [S1, self.dispmax],
            
            "B3" : [-5*N1,+5*N1],
        }
        for k,v in dict_bound.items(): dict_bound[k] = np.float64(v)
        
        prebound = self.dict_prebound
        for param in prebound.keys():
            dict_bound[param] = np.float64(prebound[param])
                
        used_bounds = [k for k, v in prebound.items() if v is not None]
        if len(used_bounds)>0:
            print(f"{self.header_printmsg} User-supplied bounds are used for {used_bounds}")
        
        preguess = self.dict_preguess
        dict_guess = {}
        
        dict_guess['V31'] = preguess['V31'] if 'V31' in preguess and preguess['V31'] is not None else V1
        dict_guess['V32'] = preguess['V32'] if 'V32' in preguess and preguess['V32'] is not None else V1
        dict_guess['V33'] = preguess['V33'] if 'V33' in preguess and preguess['V33'] is not None else V1
        dict_guess['B3']  = preguess['B3']  if 'B3'  in preguess and preguess['B3']  is not None else B1
        
        # dict_guess['A31'],dict_guess['A32'],dict_guess['A33']=np.float64([0.25,0.65,0.16])*A1
        # dict_guess['F31'],dict_guess['F32'],dict_guess['F33']=np.float64([0.5*F1,0.5*F1,0])
        # dict_guess['F31'],dict_guess['F32'],dict_guess['F33']=np.float64([0.10*F1,0.66*F1,0.30*F1])
        
        # dict_guess['F31'],dict_guess['F32'],dict_guess['F33'] = [0.10*F1,0.60*F1,0.30*F1]
        dict_guess['F31'],dict_guess['F32'],dict_guess['F33'] = [0,F1,0]
        # dict_guess['A31'],dict_guess['A32'],dict_guess['A33'] = [0.3*A1,0.6*A1,0.2*A1]
        
        
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=[0.5*S1,1.0*S1,3.0*S1]
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=np.float64([5,10,20])
        
        # guesses_sigmas = np.array([4.8,9.6,20])
        # # guesses_sigmas = np.array([6.7, 13, 20])
        # guesses_sigmas = np.sqrt(guesses_sigmas**2 + (self.chansep/2.355)**2)
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=guesses_sigmas
        
        # dict_bound['S32']=np.float64([self.dispmin,self.dispmax])
        # dict_bound['S32']=np.float64([5,15])
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=np.float64([4.2,9.6,17])
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=np.float64([0.45*S1,S1,2*S1])
        dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=np.float64([self.dispmin+0.1,S1,5*S1])
        
        # dict_guess['S31'],dict_guess['S32'],dict_guess['S33']=np.float64([4.2,9.6,17])
        
        # for param in ['A31','A32','A33','S31','S32','S33']:
        #     if param in preguess and preguess[param] is not None:
        #         if preguess[param]<dict_bound[param][0]:
        #             preguess[param] = dict_bound[param][0]+0.1
        #         elif preguess[param]>dict_bound[param][1]:
        #             preguess[param] = dict_bound[param][1]-0.1
        #         dict_guess[param] = preguess[param]
                
        # if 'S32' in preguess: dict_bound['S32']:
                
        if 'F31/F1' in preguess and preguess['F31/F1'] is not None: dict_guess['F31'] = preguess['F31/F1']*F1
        if 'F32/F1' in preguess and preguess['F32/F1'] is not None: dict_guess['F32'] = preguess['F32/F1']*F1
        if 'F33/F1' in preguess and preguess['F33/F1'] is not None: dict_guess['F33'] = preguess['F33/F1']*F1
        # if 'A31'    in preguess and preguess['A31/A1'] is not None: dict_guess['A31'] = preguess['A31/A1']*A1
        # if 'A32'    in preguess and preguess['A32/A1'] is not None: dict_guess['A32'] = preguess['A32/A1']*A1
        # if 'A33'    in preguess and preguess['A33/A1'] is not None: dict_guess['A33'] = preguess['A33/A1']*A1
        if 'S31'    in preguess and preguess['S31']    is not None: dict_guess['S31'] = preguess['S31']
        if 'S32'    in preguess and preguess['S32']    is not None: dict_guess['S32'] = preguess['S32']
        if 'S33'    in preguess and preguess['S33']    is not None: dict_guess['S33'] = preguess['S33']

        used_guesses = [k for k, v in preguess.items() if v is not None]
        if len(used_guesses)>0:
            print(f"{self.header_printmsg} User-supplied guesses are used for {used_guesses}")
            
        # dict_guess['S33'] = np.clip(dict_guess['S33'], 0,dict_bound['S33'][1]-0.1)
            
        # dict_bound['S31'][1] = dict_guess['S32']-self.chansep/2.355
        # dict_bound['S33'][0] = dict_guess['S32']+self.chansep/2.355
        
        # print(dict_bound['S33'])
        
        fitB3 = self.fit_params_3G['B3']
        fixB3 = fitB3 in ('B1','BB')    
        names_param = ['F31','F32','F33','V31','S31','S32','S33']
        # names_param = ['A31','A32','A33','V31','S31','S32','S33']
        if not fixB3: names_param += ['B3']
            
        gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)
        guess  = np.float64([dict_guess[p] for p in names_param])
        
        if fixB3:
            BB = B1 if fitB3=='B1' else self.BB
            gmodel.BB = np.float64(BB)
            lp,lp_resample = gmodel.make_log_prob_3G_V32xV33xB3x()
        else:
            lp,lp_resample = gmodel.make_log_prob_3G_V32xV33x()
                    
        gmodel.log_prob = lp
        gmodel.log_prob_resample = lp_resample
                
        return gmodel,guess
    
    # @profile
    def makefit_minimize(self, G, y=None, guess=None, save_df=False, save_self_gmodel=False):
        
        if   G==1: gmodel,guess = self.prep_1GFIT()
        elif G==2: gmodel,guess = self.prep_2GFIT()
        elif G==3: gmodel,guess = self.prep_3GFIT()
        else: raise TypeError("[Plotfit] Inappropriate G input")
        
        if save_self_gmodel:
             self.gmodel = gmodel
        # self.gmodel = gmodel
        # self.guess  = guess
        
        if     y is not None: gmodel.y=y
        if guess is not None: guess = guess
        
        guess = map_params(guess, gmodel, mode='x->u')
        res   = minimize(gmodel.log_prob_guess, x0=guess, method=self.method_minimize)#, tol=tol)#, options=dict(xatol=tol,fatol=tol))#, tol=1e-6)
        
        res_demapped = map_params(res.x, gmodel, mode='u->x')
                
        if save_df:
            names = gmodel.names_param
            
            for i, label in enumerate(names):
                self.df[label] = res_demapped[i]
            
            if 'S1'  in names: 
                G=1
                if 'BB' in self.df:
                    self.df['B1'] = self.BB
                
                F1,S1 = self.df.loc[0,['F1','S1']]    
                self.df['A1'] = F1/(S1*np.sqrt(2*np.pi))
                
            if 'S21' in names: 
                G=2
                if not np.isin('V21', names): self.df['V21'] = self.df['V1'][0]
                if not np.isin('V22', names): self.df['V22'] = self.df['V21'][0]
                if not np.isin('B2', names):  self.df['B2']  = self.df['B1'][0]
                
                F21,F22,S21,S22 = self.df.loc[0,['F21','F22','S21','S22']]
                self.df['A21'] = F21/(S21*np.sqrt(2*np.pi))
                self.df['A22'] = F22/(S21*np.sqrt(2*np.pi))
                
            if 'S31' in names: 
                G=3
                if not np.isin('V31', names): self.df['V31'] = self.df[ 'V1'][0]
                if not np.isin('V32', names): self.df['V32'] = self.df['V31'][0]
                if not np.isin('V33', names): self.df['V33'] = self.df['V31'][0]
                if not np.isin('S32', names): self.df['S32'] = self.df['S1'][0]
                if not np.isin('B3',  names): 
                    if 'BB' in self.df:
                        self.df['B3'] = self.df['BB'][0]
                    else:
                        self.df['B3'] = self.df['B1'][0]
                        
                F31,F32,F33,S31,S32,S33 = self.df.loc[0,['F31','F32','F33','S31','S32','S33']]
                self.df['A31'] = F31/(S31*np.sqrt(2*np.pi))
                self.df['A32'] = F32/(S32*np.sqrt(2*np.pi))
                self.df['A33'] = F33/(S33*np.sqrt(2*np.pi))
            
            residuals        = self.get_residuals(G=G)
            self.df[f"N{G}"] = float(np.sqrt(np.mean(residuals**2)))
            if G==1:
                A1,N1 = self.df.loc[0,['A1','N1']]
                self.df['SNR1'] = A1/N1
            else:
                As = self.df.loc[0,[f'A{G}{g}' for g in range(1,G+1)]]
                NN = self.df.loc[0,f'N{G}']
                self.df[f'SNR{G}'] = np.sum(As) / NN
                for g in range(1,G+1):
                    Ag = self.df.loc[0,f'A{G}{g}']
                    self.df[f'SNR{G}{g}'] = Ag / NN
        
        return res_demapped
    
    # @profile
    def makefit_emcee(self, G, maxiter, makefit_guess=False):
        
        try:
            plt.close("all")
        except Exception:
            pass

        self.resampled    = None
        self.gmodel       = None
        self.sampler      = None
        self.next_check   = 10000
        for GGG in range(2,G+1):
            self.df[ f'SNR{GGG}'] = np.nan
            self.df[   f'B{GGG}'] = np.nan
            self.df[f'e-_B{GGG}'] = np.nan
            self.df[f'e+_B{GGG}'] = np.nan
            for key in ['A','V','S']:
                for GG in range(1,GGG+1):
                    for gg in range(1,GG+1):
                        self.df[f'{key}{GG}{gg}']    = np.nan
                        self.df[f'e-_{key}{GG}{gg}'] = np.nan
                        self.df[f'e+_{key}{GG}{gg}'] = np.nan
        gc.collect()
        
        self.maxiter = maxiter if maxiter is not None else 100000
        
        if G==1: gmodel,guess = self.prep_1GFIT()
        if G==2: gmodel,guess = self.prep_2GFIT()
        if G==3: gmodel,guess = self.prep_3GFIT()
        
        self.gmodel, self.guess = gmodel, guess
        
        ndim     = len(guess)
        nwalkers = 10 * ndim
        
        if makefit_guess:
            guess = self.makefit_minimize(G,guess=guess,save_df=True)
            self.make_atlas(gmodel=gmodel)
            
        # if G==2 and not self.dict_preguess:
            
        #     A1 = self.df.loc[0,'A1']
        #     V1 = self.df.loc[0,'V1']
        #     S1 = self.df.loc[0,'S1']
        #     # with negligible wing comp
        #     guess1 = np.float64([0.0, A1, V1, 0, S1])
        #     guess2 = np.float64([ A1,0.0, V1, S1, S1*1.5])
            
        #     spns = gmodel._spn
            
        #     guess1 = clip_guess(map_params_physical_to_unconstr(guess1, gmodel), gmodel)
        #     guess2 = clip_guess(map_params_physical_to_unconstr(guess2, gmodel), gmodel)
            
        #     pos1 = guess1 + 0.01*spns * np.random.randn(nwalkers//2, ndim)
        #     pos2 = guess2 + 0.01*spns * np.random.randn(nwalkers//2, ndim)
        #     pos = np.vstack([pos1,pos2])
            
        # elif G==3 and not self.dict_preguess:
           
        #     A1 = self.df.loc[0,'A1']
        #     V1 = self.df.loc[0,'V1']
        #     S1 = self.df.loc[0,'S1']
        #     B1 = self.df.loc[0,'B1']
        #     # with negligible wing comp
        #     guess1 = np.float64([0.3*A1, 0.6*A1, 0.1*A1, V1, 5, 10, 25])
        #     guess2 = np.float64([0.3*A1, 0.7*A1, 0,      V1, 5, 10, 25])
        #     # guess3 = np.float64([0.0*A1, 0.5*A1, 0.5*A1, V1, 5, 10, 25])
            
        #     spns = gmodel._spn
            
        #     # guess1 = clip_guess(map_params_physical_to_unconstr(guess1, gmodel), gmodel)
        #     # guess2 = clip_guess(map_params_physical_to_unconstr(guess2, gmodel), gmodel)
            
        #     guess1 = map_params_physical_to_unconstr(guess1, gmodel)
        #     guess2 = map_params_physical_to_unconstr(guess2, gmodel)

        #     # guess3 = clip_guess(map_params_physical_to_unconstr(guess3, gmodel), gmodel)
            
        #     pos1 = guess1 + 0.01*1 * np.random.randn(nwalkers//2, ndim)
        #     pos2 = guess2 + 0.01*1 * np.random.randn(nwalkers//2, ndim)
        #     # pos3 = guess3 + 0.1*spns * np.random.randn(nwalkers//3, ndim)
        #     pos = np.vstack([pos1,pos2])
        
        # else:
        #     guess_unconstr = map_params_physical_to_unconstr(guess, gmodel)
            
        #     if np.any(np.isnan(guess_unconstr)): 
        #         for i, name in enumerate(gmodel.names_param):
        #             bound = gmodel.dict_bound[name]
        #             inoff = 'in' if bound[0]<guess[i]<bound[1] else 'off'
        #             print(self.header_printmsg, name, bound, guess[i], inoff)
        #         raise ValueError
            
        #     spns = gmodel._spn
        #     guess = clip_guess(guess_unconstr, gmodel)
            
        #     pos = guess + 0.1*spns * np.random.randn(nwalkers, ndim)
        #     # pos = np.clip(pos, -7.999, 7.999)
            
        # if G==3 and not self.dict_preguess:
        #     F1 = self.df.loc[0,'F1']
        #     V1 = self.df.loc[0,'V1']
        #     S1 = self.df.loc[0,'S1']
        #     B1 = self.df.loc[0,'B1']
        #     # with negligible wing comp
        #     guess1 = np.float64([0.3*F1, 0.5*F1, 0.2*F1, V1, 5, 10, 25])
        #     guess2 = np.float64([0.4*F1, 0.6*F1, 0.1*F1, V1, 5, 10, 25])
            
        #     if 'B3' in gmodel.names_param:
        #         guess1 = np.float64([0.3*F1, 0.5*F1, 0.2*F1, V1, 5, 10, 25, 0])
        #         guess2 = np.float64([0.4*F1, 0.6*F1, 0.1*F1, V1, 5, 10, 25, 0])
            
        #     # guess3 = np.float64([0.0*A1, 0.5*A1, 0.5*A1, V1, 5, 10, 25])
            
            
        #     # guess1 = clip_guess(map_params_physical_to_unconstr(guess1, gmodel), gmodel)
        #     # guess2 = clip_guess(map_params_physical_to_unconstr(guess2, gmodel), gmodel)
            
        #     guess1 = map_params_physical_to_unconstr(guess1, gmodel)
        #     guess2 = map_params_physical_to_unconstr(guess2, gmodel)

        #     # guess3 = clip_guess(map_params_physical_to_unconstr(guess3, gmodel), gmodel)
            
        #     pos1 = guess1 + 0.1*1 * np.random.randn(nwalkers//2, ndim)
        #     pos2 = guess2 + 0.1*1 * np.random.randn(nwalkers//2, ndim)
        #     # pos3 = guess3 + 0.1*spns * np.random.randn(nwalkers//3, ndim)
        #     pos = np.vstack([pos1,pos2])
        # else:
        
        guess_unconstr = map_params(guess, gmodel, mode='x->u')
    
        if np.any(np.isnan(guess_unconstr)): 
            for i, name in enumerate(gmodel.names_param):
                bound = gmodel.dict_bound[name]
                inoff = 'in' if bound[0]<guess[i]<bound[1] else 'off'
                print(self.header_printmsg, name, bound, guess[i], inoff)
            raise ValueError
        
        # guess = np.clip(guess_unconstr, 0.01,0.99)
        
        # guess = clip_guess(guess_unconstr, gmodel)
        pos = guess_unconstr + 0.01 * np.random.randn(nwalkers, ndim)
        pos = np.clip(pos, 0.001,0.999)
                
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            gmodel.log_prob,
            moves = [
            #     (emcee.moves.StretchMove(a=2.0), 0.5),
            #     (emcee.moves.DEMove(),           0.4),
            #     (emcee.moves.DESnookerMove(),    0.1),
                (emcee.moves.DEMove(),        0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
            
        )
        self._reset_autocorr_buffers(maxiter)
        
        for state in sampler.sample(pos, iterations=maxiter):
            if sampler.iteration < self.next_check:
                continue
            if self.check_converged(sampler, gmodel, generate_plot=self.plot_autocorr):
                break
                    
        # plot_walks_A3_S3(
        #     sampler=sampler,
        #     gmodel=gmodel,
        #     demap_func=map_params_unconstr_to_physical,
        #     outpath=self.path_plot / f"walks_{self.name_cube}{self.suffix}.png",
        #     logx=True,
        # )
            
        # if os.path.exists(self.path_plot/f'walks_{self.name_cube}{self.suffix}.png'):
            # os.remove(self.path_plot/f'walks_{self.name_cube}{self.suffix}.png')
        
        if self.plot_autocorr and self.savename_autocorr is not None and self.savename_autocorr.exists():
            try:
                os.remove(self.savename_autocorr)
            except Exception:
                pass
            
        self.fill_df_emcee(sampler,gmodel)
        self.writestat(f"{self.stat} - Done")
        self.good_NGFIT = True
        
        # self.make_atlas()
        
        try: self.make_atlas()
        except ValueError: 
            print(self.df)
            raise

        # sampler.reset()   # clears chain buffers
        del sampler
        gc.collect()
        



    # -----------------------------
    # Resampling
    # -----------------------------
    
    def resample(self, G: int, nsample: int = 1499, pbar_resample: bool = False) -> None:
        
        self.writestat(f'{self.stat} - Prep')
                
        del self.df_params
        self._init_params_df()
        
        def half_sample_mode(samples):
            """
            Computes the Half-Sample Mode of a 1D array.
            """
            samples = np.sort(np.asarray(samples))
            
            while len(samples) > 3:
                N = len(samples)
                half_idx = N // 2
                # Find the shortest interval containing half the samples
                # Interval i is [samples[i], samples[i + half_idx]]
                widths = samples[half_idx:] - samples[:N - half_idx]
                best_start = np.argmin(widths)
                
                # Keep only the samples within that shortest interval
                samples = samples[best_start : best_start + half_idx]
                
            return np.mean(samples)
        
        def fill_df_resampled(resampled, statistics=self.statistics):
            
            def calc_tension(p1,p2):
                D = p1-p2
                D = sort_outliers(D)
                med = np.median(D)
                std = np.std(D)
                return np.abs(med)/std
            
            def error_lower_upper(samples):
                p16,p50,p84 = np.percentile(samples, [16, 50, 84])
                err_minus = p50-p16
                err_plus  = p84-p50
                return err_minus,err_plus
            
            names = self.gmodel.names_param
            npars = len(names)
            
            if self.truth_from_resampling:
                for i,label in enumerate(names):
                    samples = sort_outliers(resampled[:,i])
                    p16,p50,p84 = np.percentile(samples, [16, 50, 84])
                    
                    if statistics=='MAP':
                        kde = gaussian_kde(samples)
                        x_grid = np.linspace(samples.min(),samples.max(),1000)
                        kde_values = kde.evaluate(x_grid)
                        marginal_map = x_grid[np.argmax(kde_values)]
                        # marginal_map = half_sample_mode(samples)
                        self.df[label] = marginal_map
                        
                        # idx_MLE = np.argmin(funs)
                        # self.df[label] = resampled[idx_MLE,i]

                    if statistics == 'MEDIAN':
                        self.df[label] = p50
                        
                for i, label in enumerate(names):
                    par,Gg = label[0],label[1:]
                    if par=='F':
                        FF = self.df.loc[0,label]
                        SS = self.df.loc[0,f'S{Gg}']
                        AA = FF/(SS*2.50662827463)
                        self.df[f'A{Gg}'] = AA
                    if par=='A':
                        AA = self.df.loc[0,label]
                        SS = self.df.loc[0,f'S{Gg}']
                        FF = AA*SS*2.50662827463
                        self.df[f'F{Gg}'] = FF
                        
                S1s = resampled_1G[:, 2]
                self.df['S1'] = np.percentile(sort_outliers(S1s),50)

            for ii in range(npars):
                for jj in range(ii+1,npars):
                    # colname = f'dcor_{names[ii]}{names[jj]}'
                    # dcorr = dcor_trimmed(resampled[:, ii], resampled[:, jj])
                    # self.df_params[colname] = dcorr
                    xx = resampled[:, ii]
                    yy = resampled[:, jj]
                    xx,yy = trim_outlier_mahalanobis(xx,yy)
                    self.df_params[f'rs_{names[ii]}{names[jj]}'],self.df_params[f'p_rs_{names[ii]}{names[jj]}'] = spearmanr(xx,yy)
                    
            # self.df_params['WAIC'] = compute_waic(-1*funs)
            self.df_params['AIC'],self.df_params['BIC'],self.df_params['AICc'] = compute_aic_bic_aicc((-1*funs).max(), npars, len(self.gmodel.x))

            S1s = resampled_1G[:, 2]
            
            self.df_params['S1']   = self.df.loc[0,'S1']
            # self.df_params["e_S1"] = np.nanstd(sort_outliers(S1s))
            self.df_params['e-_S1'],self.df_params['e+_S1'] = error_lower_upper(S1s)
            
            if G==1: return
            
            if G==2:
                self.df_params['sn'] = self.df.loc[0,'S21']
                self.df_params['sb'] = self.df.loc[0,'S22']
                self.df_params['An'] = gaussian_area(self.df.loc[0,'A21'],self.df.loc[0,'S21'])
                self.df_params['Ab'] = gaussian_area(self.df.loc[0,'A22'],self.df.loc[0,'S22'])
                
                iS21,iS22 = idx(names,'S21'),idx(names,'S22')
                sns = resampled[:, iS21]
                sbs = resampled[:, iS22]
                
                self.df_params['tension_S21S22'] = calc_tension(resampled[:,iS21],resampled[:,iS22])
                
            if G==3:
                self.df_params['sn'] = self.df.loc[0,'S31']
                self.df_params['sb'] = self.df.loc[0,'S32']
                self.df_params['sw'] = self.df.loc[0,'S33']
                self.df_params['An'] = gaussian_area(self.df.loc[0,'A31'],self.df.loc[0,'S31'])
                self.df_params['Ab'] = gaussian_area(self.df.loc[0,'A32'],self.df.loc[0,'S32'])
                
                iA21,iA22,iS31,iS32 = idx(names,['A31','A32','S31','S32'])
                sns = resampled[:, iS31]
                sbs = resampled[:, iS32] if 'S32' in names else self.df.loc[0,'S22']
                
                iS33 = idx(names,'S33')
                sws  = resampled[:,iS33]
                # self.df_params['e_sw'] = np.nanstd(sort_outliers(sws))
                self.df_params['e-_sw'],self.df_params['e+_sw'] = error_lower_upper(sws)
                
                self.df_params['tension_S31S32'] = calc_tension(resampled[:,iS31],resampled[:,iS32])
                self.df_params['tension_S32S33'] = calc_tension(resampled[:,iS32],resampled[:,iS33])
            
            self.df_params['At']         = self.df_params.loc[0,'An']+self.df_params.loc[0,'Ab']
            self.df_params['sn/sb']      = self.df_params.loc[0,'sn']/self.df_params.loc[0,'sb']
            self.df_params['An/At']      = self.df_params.loc[0,'An']/self.df_params.loc[0,'At']
            self.df_params['log(sb-sn)'] = np.log10(self.df_params.loc[0,'sb']-self.df_params.loc[0,'sn'])
            
            # Ans = gaussian_area(resampled[:, iA21], sns)
            # Abs = gaussian_area(resampled[:, iA22], sbs)
            # Ats = Ans + Abs
            
            
            # self.df_params["e_sn"] = np.nanstd(sort_outliers(sns))
            # self.df_params["e_sb"] = np.nanstd(sort_outliers(sbs))
            
            self.df_params['e-_sn'],self.df_params['e+_sn'] = error_lower_upper(sns)
            self.df_params['e-_sb'],self.df_params['e+_sb'] = error_lower_upper(sbs)
            
            # self.df_params["e_An"] = np.nanstd(sort_outliers(Ans))
            # self.df_params["e_Ab"] = np.nanstd(sort_outliers(Abs))
            # self.df_params["e_At"] = np.nanstd(sort_outliers(Ats))
            
            # self.df_params["e_sn/sb"]      = np.nanstd(sort_outliers(sns/sbs))
            # self.df_params["e_An/At"]      = np.nanstd(sort_outliers(Ans/Ats))
            # self.df_params["e_log(sb-sn)"] = np.nanstd(sort_outliers(np.log10(sbs - sns)))
        
            # if G==2:
                # if self.df_params.loc[0,'sb']>np.percentile(self.list_disp,90):
                #     self.df_params[  'sw'] = self.df_params.loc[0,  'sb']
                #     self.df_params['e_sw'] = self.df_params.loc[0,'e_sb']
                #     self.df_params[  'sb'] = np.nan
                #     self.df_params['e_sb'] = np.nan
        
            # self.df_params['Nsample'] = nsample
                
            return
        
        xx, e_y = self.x, self.e_y
        method_minimize = self.method_minimize
        
        gmodel_resample_1G = copy.deepcopy(self.gmodel_1G)
        names_1G = gmodel_resample_1G.names_param        
        # gmodel_resample_1G,_ = self.prep_1GFIT()
        
        gmodel_resample_XG = copy.deepcopy(self.gmodel)
        residuals_orig  = self.get_residuals(G=G)
        
        SNRG = self.df.loc[0,f'SNR{G}']
        if not np.isfinite(SNRG): return
        
        self.df_params[f"SNR{G}"] = SNRG
        self.stat = f"RES{G}G"
        self.writestat(f'{self.stat} - Start')

        names_3G = np.array(gmodel_resample_XG.names_param)
        npars_3G = len(names_3G)
                
        guess_1G = self.df.loc[0, names_1G].to_numpy(dtype=np.float64)
        guess_1G = map_params(guess_1G, gmodel_resample_1G, mode='x->u')
        
        # guess    = np.float64([self.df[label][0] for label in names])
        # guess = clip_guess(guess, gmodel_resample_XG)
        # guess = map_params_physical_to_unconstr(guess,gmodel_resample_XG)
        # if np.any(np.isnan(guess)):
        #     raise ValueError(f"{self.header_printmsg} NaN in guess parameters for resampling.")s

        flat_samples = self.sampler.get_chain(discard=self.burnin,thin=self.thin,flat=True)
        # rdm_indices  = np.random.choice(flat_samples.shape[0], nsample, replace=True)
        nflats = flat_samples.shape[0]
            
        resampled    = np.full((nsample, npars_3G), np.nan, dtype=float)
        resampled_1G = np.full((nsample, len(names_1G)), np.nan, dtype=float)

        pbar = tqdm(total=nsample) if pbar_resample else None
        timei = time.time()
        
        yy   = self.y
        leny = len(yy)
        noise_matrix = np.random.choice(residuals_orig, size=(nsample, leny), replace=True)
        
        V1,S1,B1 = self.df.loc[0,['V1','S1','B1']]
        
        fitB1 = self.fit_params_1G['B1']
                
        if fitB1 == 'BB':
            def iter_resample(y_w_noise, guess_XG):
                BB = makefit_bg_linefree_njit(xx, y_w_noise, e_y, V1, S1, B1)
                gmodel_resample_1G.BB = BB
                gmodel_resample_XG.BB = BB
                res_1G = minimize(gmodel_resample_1G.log_prob_resample, guess_1G, args=(y_w_noise), method=method_minimize)
                res_XG = minimize(gmodel_resample_XG.log_prob_resample, guess_XG, args=(y_w_noise), method=method_minimize)
                return res_1G, res_XG

        elif fitB1 == 'B1':
            self.fit_params_1G['B1']='free'
            gmodel_1G_Bfree,_ = self.prep_1GFIT()
            guess_1G_Bfree    = self.df.loc[0,['F1','V1','S1','B1']]
            def iter_resample(y_w_noise, guess_XG):
                res_1G_Bfree = minimize(   gmodel_1G_Bfree.log_prob_resample, guess_1G_Bfree, args=(y_w_noise,), method=method_minimize)
                B1 = res_1G_Bfree.x[3]
                gmodel_resample_1G.BB = B1
                gmodel_resample_XG.BB = B1
                res_1G       = minimize(gmodel_resample_1G.log_prob_resample, guess_1G, args=(y_w_noise), method=method_minimize)
                res_XG       = minimize(gmodel_resample_XG.log_prob_resample, guess_XG, args=(y_w_noise), method=method_minimize)
                return res_1G, res_XG
            self.fit_params_1G['B1']='B1'

        else:  # 'free'
            def iter_resample(y_w_noise, guess_XG):
                res_1G = minimize(gmodel_resample_1G.log_prob_resample, guess_1G, args=(y_w_noise), method=method_minimize)
                res_XG = minimize(gmodel_resample_XG.log_prob_resample, guess_XG, args=(y_w_noise), method=method_minimize)
                return res_1G, res_XG
        
        funs = np.zeros(nsample)
        BBs  = np.zeros(nsample)
        self.df_params['Nsample'] = 5000
        
        for j in range(nsample):
            while True:
                y_w_noise = yy + noise_matrix[j,:]
                rdm_index = np.random.choice(nflats)
                
                guess_XG = flat_samples[rdm_index,:]
                
                res_1G,res_XG = iter_resample(y_w_noise, guess_XG)
                if G==1: break
                
                if np.isfinite(res_XG.fun):
                    resampled_1G[j, :] = res_1G.x
                    resampled[   j, :] = res_XG.x
                    funs[j] = res_XG.fun
                    BBs[j]  = gmodel_resample_XG.BB
                    break
                                
            if j % 500 == 0 and j != 0:
                timef = time.time()
                eta = datetime.timedelta(seconds=int((nsample - j) / (500 / (timef - timei))))
                self.writestat(f"{self.stat} {j} {eta}")
                
                if j>1000:
                    converged = check_converged_resampling(resampled[:j,:], j, npars_3G)
                    if converged: 
                        resampled    = resampled[   :j,:]
                        resampled_1G = resampled_1G[:j,:]
                        BBs       = BBs[:j]
                        funs      = funs[:j]
                        self.df_params['Nsample'] = j
                        break
                
                timei = timef

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()
            
        
        # if np.all((resampled - guess[None,:])==0):
        #     print(self.header_printmsg)
            
        resampled    = map_params(resampled,self.gmodel,mode='u->x')
        resampled_1G = map_params(resampled_1G,gmodel_resample_1G,mode='u->x') 
        fill_df_resampled(resampled, self.statistics)
        if fitB1=='BB' or fitB1=='B1':
            resampled = np.column_stack((resampled,BBs))
        
        self.resampled = resampled
        np.save('/home/mskim/workspace/research/resampled.npy',resampled)
        
        self.writestat(f'{self.stat} - Done')

        self.good_resample = True
        self.df_params["Reliable"] = "Y"
        self.make_atlas(cleanup=True)
        
        return


    # -----------------------------
    # Checks / guards
    # -----------------------------
    def check_config(self) -> None:
        # Enforce allowed values
        self.fit_params_2G["A21"] = "free"
        self.fit_params_2G["A22"] = "free"
        self.fit_params_2G["S21"] = "free"
        self.fit_params_2G["S22"] = "free"

        if self.fit_params_2G["V21"] == 0:
            self.fit_params_2G["V21"] = "0"

        if self.fit_params_2G["V21"] not in ["0", "free", "V1"]:
            raise TypeError(f'{self.header_printmsg} V21 must be one of {"0","free","V1"}.')
        if self.fit_params_2G["V22"] not in ["0", "free", "V21"]:
            raise TypeError(f'{self.header_printmsg} V22 must be one of {"0","free","V21"}.')
        if self.fit_params_2G["B2"] not in ["free", "fix", "B1"]:
            raise TypeError(f'{self.header_printmsg} B2 must be one of {"free","fix","B1"}.')

        if self.fit_params_2G["V21"] == "0" and self.fit_params_2G["V22"] == "free":
            print(f"{self.header_printmsg} Setting V22=V21; V22 free not possible when V21==0")
            self.fit_params_2G["V22"] = "V21"
        if self.fit_params_2G["V21"] == "V1" and self.fit_params_2G["V22"] == "free":
            print(f"{self.header_printmsg} Setting V22=V21; V22 free not possible when V21==V1")
            self.fit_params_2G["V22"] = "V21"

    def check_stacked(self) -> bool:
        if np.all(self.y == 0) or np.all(~np.isfinite(self.y)):
            self.df_params["Reliable"] = "0stacked"
            # print(f"{self.header_printmsg} {self.name_cube} pass; Nothing seems to be stacked.")
            return False

        self.xmin, self.xmax = map(float, np.nanpercentile(self.x, [0, 100]))
        self.ymin, self.ymax = map(float, np.nanpercentile(self.y, [0, 100]))
        self.chansep         = float(np.abs(np.mean(np.diff(self.x))))
        self.bandwidth       = float(self.xmax - self.xmin)
        
        return True

    def check_list_disp(self) -> bool:
        # Bounds from spectral resolution & bandwidth
        self.dispmin = (self.chansep / 2.355)
        # self.dispmin = 2.0
        # self.dispmax = 999 #self.bandwidth / 2.355
        self.dispmax = 99 #self.bandwidth / 2.355

        if (self.dispmax - self.dispmin) < 2:
            self.df_params["Reliable"] = "disprng"
            print(f"{self.header_printmsg} pass; Range of stacked dispersion is narrow")
            return False
        return True

    # -----------------------------
    # Post-fit evaluations
    # -----------------------------
    def evaluate_1GFIT(self) -> None:
        
        self.good_1GFIT = True
        A1, V1, S1, B1, SNR1 = self.df.loc[0, ["A1", "V1", "S1", "B1", "SNR1"]]

        if SNR1<5:
            if self.verbose:
                print(f"{self.header_printmsg} 2GFIT no-go; SNR1<5")
            self.df_params["Reliable"] = "lowSNR1"
            self.good_1GFIT = False
            return
        
        if S1 * 2.3 * 2 > self.bandwidth:
            if self.verbose:
                print(f"{self.header_printmsg} 2GFIT no-go; Bandwidth too narrow")
            self.df_params["Reliable"] = "W_S1>BW"
            self.good_1GFIT = False
            return
        
        if SNR1<10:
            # print(f"{self.header_printmsg} {self.name_cube} Forcing B2=B1; SNR1<15")
            self.fit_params_2G['B2']='B1'
            self.fit_params_3G['B3']='B1'
            
    def evaluate_2GFIT(self) -> None:
        
        redo_2GFIT = False
        goto_1GFIT = False
        
        N1,A1,S1,B1   = self.df.loc[0,['N1','A1','S1','B1']]
        A22,S22,N2,B2 = self.df.loc[0,['A22','S22','N2','B2']]
        
        if self.fit_params_2G['B2']=='free':
            if B2<-3*N1 or B2>3*N1:
                self.fit_params_1G['B1']='B1'
                self.fit_params_2G['B2']='B1'
                self.fit_params_3G['B3']='B1'
                redo_2GFIT = True
            
        return redo_2GFIT, goto_1GFIT
        
    def evaluate_3GFIT(self) -> None:
        
        if self.fallback_to_2gfit == False:
            return False, False
                                
        redo_3GFIT = False
        goto_2GFIT = False
        
        F31,F32,F33 = self.df.loc[0,['F31','F32','F33']]
        A31,A32,A33 = self.df.loc[0,['A31','A32','A33']]
        # F31,F32,F33 = self.df.loc[0,['F31','F32','F33']]
        SNR3,B3,N3  = self.df.loc[0,['SNR3','B3','N3']]
        S31,S32,S33 = self.df.loc[0,['S31','S32','S33']]
        A1,S1,N1,B1 = self.df.loc[0,['A1','S1','N1','B1']]
        
        conditions = {
            # 'F32<F31': F32<F31,
            # 'F32<F33': F32<F33,
            # 'F31+F32<F33': F31+F32<F33,
            # 'A32<A33': A32<A33,
            # 'F<0': np.any(np.array([F31,F32,F33])<0),
            
            # "dcor_F32F33": self.df.loc[0,'dcor_F32F33'] > 0.995,
            # "dcor_F31F32": self.df.loc[0,'dcor_F31F32'] > 0.998,
            'ddisp_S33S32': S33-S32<1.0*self.chansep/2.355,
            'ddisp_S32S31': S32-S31<1.0*self.chansep/2.355,
            #    'negA': np.any(np.array([A31,A32,A33])<0),
            # 'S33>4S1': S33>4*S1,
        }
        # print(self.header_printmsg, F32,F33, F32<F33)

        triggered = [name for name, cond in conditions.items() if cond]

        if triggered:
            print(f"{self.header_printmsg} Switching to 2GFIT. Failed flags: {triggered}")

            self.df[f'SNR3'] = np.nan
            self.df[f'B3']   = np.nan
            self.df[f'e-_B3'] = np.nan
            self.df[f'e+_B3'] = np.nan

            for key in ['A','V','S']:
                for GG in range(1,3+1):
                    for gg in range(1,3+1):
                        self.df[f'{key}{GG}{gg}']    = np.nan
                        self.df[f'e-_{key}{GG}{gg}'] = np.nan
                        self.df[f'e+_{key}{GG}{gg}'] = np.nan

            for key in self.df_params.keys():
                if key in ['Name','suffix','Reliable']:
                    continue
                self.df_params[key] = np.nan

            self.df_params['Nsample'] = 0
            goto_2GFIT = True
            
        #====
        
        if self.fit_params_3G['B3']=='free':
            if B3<-3*N1 or B3>3*N1:
                self.fit_params_1G['B1']='B1'
                self.fit_params_2G['B2']='B1'
                self.fit_params_3G['B3']='B1'
                redo_3GFIT = True
            
        return redo_3GFIT, goto_2GFIT

        # if self.fit_params_3G['B3']=='fre':
        #     # if B3<B1-N1 or B3>B1+N1 or S33>S1*5:
        #     if S33>S1*5:
        #         self.fit_params_3G['B3']='B1'
        #         self.fit_params_2G['B2']='B1'
        #         # self.redo_3GFIT=True
        #         self.goto_2GFIT = True
        #         # return
            
        # return
        
        
        # Switch to 2GFIT
        SNR31,SNR32,SNR33 = A31/N3,A32/N3,A33/N3
        A3dA1s = np.array([A31,A32,A33])/A1
        snr3s  = np.array([SNR31,SNR32,SNR33])
        # if np.any(snr3s<1):# or np.any(A3dA1s<0.05):
        #     self.goto_2GFIT = True
        
        if SNR32<1: self.goto_2GFIT = True
    
        # if SNR33<0.5:# or A33/A1<0.05:
        #     self.goto_2GFIT = True
            
        if S33>self.bandwidth / 2.355:
            self.goto_2GFIT = True
        # if S33<np.nanpercentile(self.list_disp,99):
        #     self.redo_2GFIT = True
        
        return

    def evaluate_final(self) -> None:
        A21, A22, N2, SNR2 = self.df.loc[0, ["A21", "A22", "N2", "SNR2"]]
        if float(SNR2) < 15:
            self.df_params["Reliable"] = "lowSNR2"
            self.good_NGFIT = False
            return
        if float(A21) < 3 * float(N2):
            self.df_params["Reliable"] = "lowA21"
            self.good_NGFIT = False
            return
        if float(A22) < 3 * float(N2):
            self.df_params["Reliable"] = "lowA22"
            self.good_NGFIT = False
            return



    # -----------------------------
    # Public run method
    # -----------------------------
    
    def do_1GFIT(self):
        
        self.redo_1GFIT = True
        self.makefit_minimize(G=1,save_df=True,save_self_gmodel=True)
        self.resample(1,self.nsample_resample,self.pbar_resample)
    
    def do_2GFIT(self):
        
        self.redo_2GFIT = True
        self.goto_1GFIT = False
        
        # while self.redo_2GFIT:
        #     if self.truth_from_resampling:
        #         self.makefit_emcee(G=2,maxiter=100000,makefit_guess=False)
        #         self.resample(2,self.nsample_resample,self.pbar_resample)
        #     else:
        #         self.makefit_emcee(G=2,maxiter=100000,makefit_guess=False)
        #     self.evaluate_2GFIT()
            
        #     if self.goto_1GFIT:
        #         self.do_1GFIT()
        #         return
            
        # if not self.truth_from_resampling:
        #     self.resample(2,self.nsample_resample,self.pbar_resample)
            
        self.makefit_emcee(G=2,maxiter=50000,makefit_guess=False)
        redo_2GFIT, _ = self.evaluate_2GFIT()
        if redo_2GFIT:
            self.make_atlas(cleanup=True)
            self.makefit_emcee(G=2,maxiter=50000,makefit_guess=False)
        self.resample(2,self.nsample_resample,self.pbar_resample)
        
        return
    
    
    def do_3GFIT(self):
        
        # if self.df.loc[0,'SNR1']<6:
        #     self.do_1GFIT()
        #     return
        # if self.df.loc[0,'SNR1']<9:
        #     self.do_2GFIT()
        #     return
        
        # self.redo_3GFIT = True
        # self.goto_2GFIT = False
        
        # while self.redo_3GFIT:
        #     if self.truth_from_resampling:
        #         self.makefit_emcee(G=3,maxiter=10000,makefit_guess=False)
        #         self.resample(3,self.nsample_resample,self.pbar_resample)
        #     else:
        #         self.makefit_emcee(G=3,maxiter=50000,makefit_guess=False)
        #         # self.makefit_ptemcee(G=3, maxiter=100000, makefit_guess=False)
            
        #     # self.makefit_emcee(G=3,maxiter=200000,makefit_guess=False)
                                
        #     # self.evaluate_3GFIT()
            
        #     if self.goto_2GFIT:
        #         self.do_2GFIT()
        #         return
                    
        # if not self.truth_from_resampling:
            # self.resample(3,self.nsample_resample,self.pbar_resample)
            
        self.makefit_emcee(G=3,maxiter=100000,makefit_guess=True)
        self.writestat('STBY - .')
        redo_3GFIT, goto_2GFIT = self.evaluate_3GFIT()
        
        if redo_3GFIT:
            self.make_atlas(cleanup=True)
            self.makefit_emcee(G=3,maxiter=100000,makefit_guess=False)
            _, goto_2GFIT = self.evaluate_3GFIT()
        
        if goto_2GFIT:
            self.make_atlas(cleanup=True)
            self.do_2GFIT()
            return
        
        self.resample(3,self.nsample_resample,self.pbar_resample)
        if self.truth_from_resampling:
            redo_3GFIT, goto_2GFIT = self.evaluate_3GFIT()
            if goto_2GFIT:
                self.make_atlas(cleanup=True)
                self.do_2GFIT()
                return
        return
    
    # -----------------------------
    # Plotter wrapper
    # -----------------------------
    
    def make_atlas(self, gmodel=None, cleanup=False) -> None:
        if self.path_plot is None:
            return
        
        if gmodel      is None: gmodel = self.gmodel
        if self.gmodel is None: gmodel = self.gmodel_1G
        
        plotter = Plotter(
            self.path_plot,
            self.name_cube,
            self.suffix,
            self.list_disp,
            self.df,
            self.df_params,
            gmodel,
            self.sampler,
            self.resampled,
            self.burnin,
            self.thin,
            self.list_NHI_
        )
        
        if 'S31' in gmodel.names_param: G=3
        if 'S21' in gmodel.names_param: G=2
        if 'S1'  in gmodel.names_param: G=1
        
        # plotter.makeplot_atlas(G)
                        
        try:
            plotter.makeplot_atlas(G)
        except Exception as e:
            print(self.name_cube, self.suffix, e)
            # raise
        
        if cleanup: plotter.cleanup()
            
        del plotter
        
        plt.close('all')
        gc.collect()
        return
    

    def cleanup(self, *, aggressive: bool = True, trim_malloc: bool = True) -> None:
        """Best-effort memory cleanup for long batch runs."""
        # 1) Close matplotlib figures (this is big)
        try:
            plt.close("all")
        except Exception:
            pass

        # 2) Delete / null big attributes (only those you don't need afterwards)
        big_attrs = [
            # raw data arrays
            "x", "y", "e_y",
            "resampled", "flat_samples",

            # models can hold references to data arrays
            "gmodel", "gmodel_1G",

            # sometimes large
            "df_stacked",

            # any cached dicts that might be large
            "dict_stacked", "list_disp","sampler"
        ]

        for a in big_attrs:
            if hasattr(self, a):
                try:
                    setattr(self, a, None)
                except Exception:
                    pass

        # 3) If you created any very large local caches, clear them too
        # Example: autocorr buffers
        for a in ["autocorr_mean", "autocorr_max", "chekstep"]:
            if hasattr(self, a):
                try:
                    setattr(self, a, None)
                except Exception:
                    pass

        # 4) Force garbage collection
        gc.collect()

        # 5) Optional: on Linux, ask libc to return freed arenas to OS
        # This can reduce RSS *sometimes*.
        if aggressive and trim_malloc:
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

    
    def run(self, suffix: str = "", nsample_resample: int = 1499, pbar_resample: bool = False) -> None:
        
        self.suffix = suffix
        self.nsample_resample = int(nsample_resample)
        self.pbar_resample = pbar_resample
        self.header_printmsg = f'[Plotfit {self.name_cube}{self.suffix}]'
        
        self.stat = 'START'
        self.writestat(f'{self.stat} . .')
        
        self.check_config()
        if not self.check_stacked():
            return
        if not self.check_list_disp():
            return
        
        del self.dict_stacked
        gc.collect()

        # self.limit_range(multiplier_disp=7)
        self.limit_range(multiplier_disp=10)
        
        # self.symmeterise_x()
        
        # del self.df_stacked
        gc.collect()
        
        self.stat = '1GFIT'
        self.writestat(f'{self.stat} . .')
                
        # self.makefit_emcee(G=1,maxiter=50000)
        self.makefit_minimize(G=1,save_df=True)
        
        # self.limit_range(multiplier_disp=9)
        
        self.evaluate_1GFIT()
        self.make_atlas(gmodel=self.gmodel_1G, cleanup=True)
                
        if not self.good_1GFIT:
            self.removestat()
            return
        
        #1
        self.makefit_bg_linefree(multiplier_mask_S1=5)
        # self.fit_params_1G['B1']='free'
        # self.fit_params_2G['B2']='free'
        # self.fit_params_3G['B3']='free'
        
        self.makefit_minimize(G=1,save_df=True)
        # self.makefit_emcee(G=1,maxiter=50000)
        #2
        # self.fit_params_3G['B3'] = 'B1'
        # self.fit_params_3G['B3'] = 'fre'
        # self.fit_params_3G['V33'] = 'fre'
        # self.fit_params_3G['S32'] = 'S1'
        
        self.do_2GFIT()
        # self.do_3GFIT()
        self.removestat()
        
        self.cleanup()