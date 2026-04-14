# Expects on self: burnin, thin, old_tau, old_tau_max, multiplier_burnin_maxtau,
#                  slope_tau, testlength, maxiter, next_check, plot_autocorr,
#                  path_temp, path_plot, name_cube, suffix, stat,
#                  autocorr_mean, autocorr_max, chekstep, savename_autocorr,
#                  autocorr_argmax, _timer_start, _last_iter
from __future__ import annotations

import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..gmodel import Gmodel, map_params


class DiagnosticsMixin:

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
    # Parameter diagnostics
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

    # -----------------------------
    # Autocorr buffers / plots
    # -----------------------------
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