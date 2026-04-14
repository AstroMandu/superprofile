# Expects on self: df, df_params, gmodel, sampler, burnin, thin, statistics,
#                  fit_params_1G/2G/3G, fallback_to_2gfit, chansep, bandwidth,
#                  list_disp, header_printmsg, verbose, BB
from __future__ import annotations

import gc

import emcee
import numpy as np
from scipy.stats import gaussian_kde, spearmanr

from ..gmodel import Gmodel, map_params
from ..subroutines import (gaussian_area, idx, sort_outliers,
                           trim_outlier_mahalanobis, compute_aic_bic_aicc,
                           compute_waic)


class EvaluationMixin:

    # -----------------------------
    # Fill df from emcee chain
    # -----------------------------
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