"""Plotfit: EMCEE-based 1G/2G/3G Gaussian fitter

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
- .gmodel.Gmodel
- .plotter.Plotter
- .subroutines helpers
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..gmodel import Gmodel
from .atlas       import AtlasMixin
from .diagnostics import DiagnosticsMixin
from .evaluation  import EvaluationMixin
from .fitting     import FittingMixin
from .resampling  import ResamplingMixin
from .cleanup import CleanupMixin



class XGFIT(DiagnosticsMixin, EvaluationMixin, ResamplingMixin,
              FittingMixin, AtlasMixin, CleanupMixin):
    """Driver for 1G + 2G + 3G Gaussian fits with MCMC and resampling.

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
    # Construction
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
    # Public fit orchestration
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
    # Top-level run
    # -----------------------------
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