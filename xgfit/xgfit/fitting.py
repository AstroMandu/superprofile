# Expects on self: x, y, e_y, df, df_params, chansep, xmin, xmax, ymin, ymax,
#                  dispmin, dispmax, bandwidth, list_disp, list_NHI_,
#                  fit_params_1G/2G/3G, dict_preguess, dict_prebound,
#                  method_minimize, gmodel, gmodel_1G, guess, guess_1G,
#                  BB, burnin, thin, maxiter, next_check, slope_tau,
#                  testlength, plot_autocorr, header_printmsg, stat, suffix
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
from typing import Iterable, Tuple

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..gmodel import Gmodel, map_params
from ..subroutines import gauss, idx


class FittingMixin:

    # -----------------------------
    # 1G quick minimize (used by limit_range)
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

    # -----------------------------
    # Spectral range trimming
    # -----------------------------
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

    # -----------------------------
    # Residuals
    # -----------------------------
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

    # -----------------------------
    # Background pre-fit
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

    # -----------------------------
    # Gmodel prep: 1G / 2G / 3G
    # -----------------------------
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

    # -----------------------------
    # Minimize-based fit
    # -----------------------------
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
                self.df['A22'] = F22/(S22*np.sqrt(2*np.pi))
                
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

    # -----------------------------
    # MCMC fit
    # -----------------------------
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