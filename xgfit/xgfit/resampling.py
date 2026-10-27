# Expects on self: x, y, e_y, df, df_params, gmodel, gmodel_1G, sampler,
#                  burnin, thin, statistics, fit_params_1G, truth_from_resampling,
#                  method_minimize, BB, header_printmsg, stat, suffix,
#                  list_disp, chansep
from __future__ import annotations

import copy
import datetime
import gc
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, spearmanr
from tqdm import tqdm

from ..gmodel import map_params
from ..subroutines import (check_converged_resampling, compute_widths, gaussian_area, idx,
                           makefit_bg_linefree_njit, sort_outliers,
                           trim_outlier_mahalanobis, compute_aic_bic_aicc,gauss, kde_map,
                           fill_snr)


def synchronize_tied_centers(frame, names, components):
    """Refresh derived centers after adopting resampled free parameters."""
    if components < 2:
        return
    shared = f'V{components}1'
    if shared not in names:
        frame[shared] = frame['V1']
    for component in range(2, components + 1):
        center = f'V{components}{component}'
        if center not in names:
            frame[center] = frame[shared]


def fit_resampled_baseline(model, physical_guess, data, method):
    """Fit in mapped coordinates and return the baseline in physical units."""
    mapped = map_params(physical_guess, model, mode='x->u')
    result = minimize(model.log_prob_resample, mapped, args=(data,), method=method)
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Resampled baseline fit failed to converge')
    physical = map_params(result.x, model, mode='u->x')
    baseline = float(physical[idx(model.names_param, 'B1')])
    if not np.isfinite(baseline):
        raise RuntimeError('Resampled baseline fit returned a nonfinite baseline')
    return baseline


def fill_correlation_diagnostics(frame, names, samples, enabled=False):
    """Optional robust pairwise diagnostics; preserve columns when disabled."""
    for ii in range(len(names)):
        for jj in range(ii + 1, len(names)):
            correlation, probability = np.nan, np.nan
            if enabled:
                xx, yy = trim_outlier_mahalanobis(samples[:, ii], samples[:, jj])
                correlation, probability = spearmanr(xx, yy)
            frame[f'rs_{names[ii]}{names[jj]}'] = correlation
            frame[f'p_rs_{names[ii]}{names[jj]}'] = probability


def refresh_fit_diagnostics(fitter, names, components):
    """Update noise and S/N from the final, consistent model parameters."""
    synchronize_tied_centers(fitter.df, names, components)
    for count in sorted({1, components}):
        fill_snr(fitter.df, fitter.get_residuals(G=count), count)


def residual_noise_rows(residuals, nsample, channels, batch_size=128):
    """Bound peak noise memory while preserving legacy draws and RNG state.

    Advance the global RNG as the old full-matrix draw did. Replay those draws
    from a private copy in small batches, leaving optimizer/start draws intact.
    """
    replay = np.random.RandomState()
    replay.set_state(np.random.get_state())
    for start in range(0, nsample, batch_size):
        np.random.choice(residuals, size=(min(batch_size, nsample-start), channels), replace=True)

    def rows():
        for start in range(0, nsample, batch_size):
            batch = replay.choice(residuals, size=(min(batch_size, nsample-start), channels), replace=True)
            yield from batch
    return rows()


class ResamplingMixin:

    def resample(self, G: int, nsample: int = 1499, pbar_resample: bool = False) -> None:
        
        self.writestat(f'{self.stat} - Prep')
                
        del self.df_params
        self._init_params_df()
        
        # def fill_SNR(G):
        #     residuals        = self.get_residuals(G=G)
        #     self.df[f"N{G}"] = float(np.sqrt(np.mean(residuals**2)))
        #     if G==1:
        #         A1,N1 = self.df.loc[0,['A1','N1']]
        #         self.df['SNR1'] = A1/N1
        #     else:
        #         As = self.df.loc[0,[f'A{G}{g}' for g in range(1,G+1)]]
                                
        #         NN = self.df.loc[0,f'N{G}']
        #         self.df[f'SNR{G}'] = np.sum(As) / NN
                
        #         for g in range(1,G+1):
        #             Ag = self.df.loc[0,f'A{G}{g}']
        #             self.df[f'SNR{G}{g}'] = Ag / NN
        #     return
        
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
                
                # XG parameters (existing)
                for i, label in enumerate(names):
                    samples = sort_outliers(resampled[:,i])
                    p16,p50,p84 = np.percentile(samples, [16, 50, 84])
                    
                    if statistics=='MAP':
                        map_val, x_grid, cdf, imax = kde_map(samples, with_cdf=True)
                        self.df[label] = map_val
                        map_cdf = cdf[imax]
                        e_plus  = np.interp(np.clip(map_cdf + 0.341, 0, 1), cdf, x_grid) - map_val
                        e_minus = map_val - np.interp(np.clip(map_cdf - 0.341, 0, 1), cdf, x_grid)
                        self.df[f'e+_{label}'] = e_plus
                        self.df[f'e-_{label}'] = e_minus
                    if statistics == 'MEDIAN':
                        self.df[label] = p50
                        self.df[f'e+_{label}'] = p84-p50
                        self.df[f'e-_{label}'] = p50-p16

                # F -> A conversion (existing)
                for i, label in enumerate(names):
                    par,Gg = label[0],label[1:]
                    if par=='F':
                        FF = self.df.loc[0,label]
                        SS = self.df.loc[0,f'S{Gg}']
                        AA = FF/(SS*2.50662827463)
                        self.df[f'A{Gg}'] = AA

                # 1G parameters from resampled_1G (add)
                def adopt_samples(samples, label):
                    samples = sort_outliers(samples)
                    p16,p50,p84 = np.percentile(samples, [16, 50, 84])
                    if statistics == 'MAP':
                        map_val, x_grid, cdf, imax = kde_map(samples, with_cdf=True)
                        self.df[label] = map_val
                        map_cdf = cdf[imax]
                        e_plus  = np.interp(np.clip(map_cdf + 0.341, 0, 1), cdf, x_grid) - map_val
                        e_minus = map_val - np.interp(np.clip(map_cdf - 0.341, 0, 1), cdf, x_grid)
                        self.df[f'e+_{label}'] = e_plus
                        self.df[f'e-_{label}'] = e_minus
                    elif statistics == 'MEDIAN':
                        self.df[label]          = p50
                        self.df[f'e+_{label}']  = p84-p50
                        self.df[f'e-_{label}']  = p50-p16

                for i, label in enumerate(names_1G):
                    adopt_samples(resampled_1G[:, i], label)

                # Use the same baseline samples as the resampled objectives.
                if fitB1 in ('BB', 'B1'):
                    adopt_samples(BBs, 'B1')
                    self.BB = float(self.df.loc[0, 'B1'])
                    self.df['BB'] = self.BB
                    if G > 1 and f'B{G}' not in names:
                        self.df[f'B{G}'] = self.BB

                # Derived centers must follow the newly adopted shared center
                # before residuals, information criteria, plots and CSV output.
                synchronize_tied_centers(self.df, names, G)

                # recompute A1 from adopted F1, S1
                F1 = self.df.loc[0,'F1']
                S1 = self.df.loc[0,'S1']
                self.df['A1'] = F1 / (S1 * 2.50662827463)

                refresh_fit_diagnostics(self, names, G)
                
                n = len(self.gmodel.x)
                # G=1: residuals from adopted 1G parameters
                if G == 1:
                    residuals_1G = self.get_residuals(G=1)
                else:
                    # recompute 1G residuals from median of resampled_1G
                    S1s = resampled_1G[:, 2]
                    # use df which already has adopted 1G params filled
                    residuals_1G = self.y - (gauss(self.x, self.df.loc[0,'A1'], self.df.loc[0,'V1'], self.df.loc[0,'S1']) + self.df.loc[0,'B1'])
                
                logl_1G = -0.5 * np.sum((residuals_1G / self.e_y)**2 + np.log(2*np.pi*self.e_y**2))
                npars_1G = len(gmodel_resample_1G.names_param)
                self.df['AIC1'], self.df['BIC1'], self.df['AICc1'] = compute_aic_bic_aicc(logl_1G, npars_1G, n)

                # XG: residuals from adopted XG parameters
                residuals_XG = self.get_residuals(G=G)
                logl_XG = -0.5 * np.sum((residuals_XG / self.e_y)**2 + np.log(2*np.pi*self.e_y**2))
                self.df[f'AIC{G}'], self.df[f'BIC{G}'], self.df[f'AICc{G}'] = compute_aic_bic_aicc(logl_XG, npars, n)

            fill_correlation_diagnostics(
                self.df_params, names, resampled,
                enabled=getattr(self, 'correlation_diagnostics', False))
                    
            



            logl_1G = -0.5 * np.sum((residuals_1G / self.e_y)**2 + np.log(2*np.pi*self.e_y**2))
            npars_1G = len(gmodel_resample_1G.names_param)
            self.df_params['AIC1'], self.df_params['BIC1'], self.df_params['AICc1'] = \
                compute_aic_bic_aicc(logl_1G, npars_1G, n)

            # XG: residuals from adopted XG parameters
            residuals_XG = self.get_residuals(G=G)
            logl_XG = -0.5 * np.sum((residuals_XG / self.e_y)**2 + np.log(2*np.pi*self.e_y**2))
            self.df_params[f'AIC{G}'], self.df_params[f'BIC{G}'], self.df_params[f'AICc{G}'] = \
                compute_aic_bic_aicc(logl_XG, npars, n)

            S1s = resampled_1G[:, 2]
            
            self.df_params['S1']   = self.df.loc[0,'S1']
            # self.df_params["e_S1"] = np.nanstd(sort_outliers(S1s))
            self.df_params['e-_S1'],self.df_params['e+_S1'] = error_lower_upper(S1s)
            
            if G==1: return
            
            if G==2:
                self.df_params['sn']    = self.df.loc[0,'S21']
                self.df_params['sb']    = self.df.loc[0,'S22']
                self.df_params['sn/S1'] = self.df_params['sn']/self.df.loc[0,'S1']
                self.df_params['sb/S1'] = self.df_params['sb']/self.df.loc[0,'S1']
                self.df_params['sn/sb'] = self.df_params['sn']/self.df_params['sb']
                self.df_params['An']    = self.df.loc[0,'F21']
                self.df_params['Ab']    = self.df.loc[0,'F22']
                self.df_params['At']    = self.df.loc[0,'F21'] + self.df.loc[0,'F22']
                self.df_params['An/At'] = self.df_params.loc[0,'An']/self.df_params.loc[0,'At']
                self.df_params['Ab/At'] = self.df_params.loc[0,'Ab']/self.df_params.loc[0,'At']
                
                iS21,iS22 = idx(names,'S21'),idx(names,'S22')
                sns = resampled[:, iS21]
                sbs = resampled[:, iS22]
                
                Ans = resampled[:, idx(names,'F21')]
                Abs = resampled[:, idx(names,'F22')]
                Ats = Ans + Abs
                
                self.df_params['e-_sn'],self.df_params['e+_sn']       = error_lower_upper(sns)
                self.df_params['e-_sb'],self.df_params['e+_sb']       = error_lower_upper(sbs)
                self.df_params['e-_sn/S1'],self.df_params['e+_sn/S1'] = error_lower_upper(sns/S1s)
                self.df_params['e-_sb/S1'],self.df_params['e+_sb/S1'] = error_lower_upper(sbs/S1s)
                self.df_params['e-_sn/sb'],self.df_params['e+_sn/sb'] = error_lower_upper(sns/sbs)
                
                self.df_params['e-_An'],self.df_params['e+_An']       = error_lower_upper(Ans)
                self.df_params['e-_Ab'],self.df_params['e+_Ab']       = error_lower_upper(Abs)
                self.df_params['e-_At'],self.df_params['e+_At']       = error_lower_upper(Ats)
                self.df_params['e-_An/At'],self.df_params['e+_An/At'] = error_lower_upper(Ans/Ats)
                self.df_params['e-_Ab/At'],self.df_params['e+_Ab/At'] = error_lower_upper(Abs/Ats)

                self.df_params['tension_S21S22'] = calc_tension(resampled[:,iS21],resampled[:,iS22])
                
                self.df_params['SNR2'] = self.df.loc[0,'SNR2']
                self.df_params['SNR21'] = self.df.loc[0,'SNR21']
                self.df_params['SNR22'] = self.df.loc[0,'SNR22']
                
            if G==3:
                self.df_params['sn'] = self.df.loc[0,'S31']
                self.df_params['sb'] = self.df.loc[0,'S32']
                self.df_params['sw'] = self.df.loc[0,'S33']
                self.df_params['An'] = self.df.loc[0,'F31']
                self.df_params['Ab'] = self.df.loc[0,'F32']
                self.df_params['Aw'] = self.df.loc[0,'F33']
                self.df_params['At'] = self.df_params.loc[0,'An'] + self.df_params.loc[0,'Ab'] + self.df_params.loc[0,'Aw']
                self.df_params['An/At'] = self.df_params.loc[0,'An']/self.df_params.loc[0,'At']
                self.df_params['Ab/At'] = self.df_params.loc[0,'Ab']/self.df_params.loc[0,'At']
                self.df_params['Aw/At'] = self.df_params.loc[0,'Aw']/self.df_params.loc[0,'At']
                
                iA21,iA22,iS31,iS32 = idx(names,['A31','A32','S31','S32'])
                sns = resampled[:, iS31]
                sbs = resampled[:, iS32] if 'S32' in names else self.df.loc[0,'S22']
                iS33 = idx(names,'S33')
                sws  = resampled[:,iS33]
                # self.df_params['e_sw'] = np.nanstd(sort_outliers(sws))
                self.df_params['e-_sw'],self.df_params['e+_sw'] = error_lower_upper(sws)
                
                Ans = resampled[:, idx(names,'F31')]
                Abs = resampled[:, idx(names,'F32')]
                Aws = resampled[:, idx(names,'F33')]
                Ats = Ans + Abs + Aws
                
                self.df_params['e-_sn'],self.df_params['e+_sn'] = error_lower_upper(sns)
                self.df_params['e-_sb'],self.df_params['e+_sb'] = error_lower_upper(sbs)
                self.df_params['e-_sw'],self.df_params['e+_sw'] = error_lower_upper(sws)
                
                self.df_params['e-_An'],self.df_params['e+_An'] = error_lower_upper(Ans)
                self.df_params['e-_Ab'],self.df_params['e+_Ab'] = error_lower_upper(Abs)
                self.df_params['e-_At'],self.df_params['e+_At'] = error_lower_upper(Ats)
                self.df_params['e-_An/At'],self.df_params['e+_An/At'] = error_lower_upper(Ans/Ats)
                self.df_params['e-_Ab/At'],self.df_params['e+_Ab/At'] = error_lower_upper(Abs/Ats)
                self.df_params['e-_Aw/At'],self.df_params['e+_Aw/At'] = error_lower_upper(Aws/Ats)

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
        #     raise ValueError(f"{self.header_printmsg} NaN in guess parameters for resampling.")

        chain_samples = self.sampler.get_chain(discard=self.burnin,thin=self.thin,flat=False)
        # rdm_indices  = np.random.choice(flat_samples.shape[0], nsample, replace=True)
        chain_walkers = chain_samples.shape[1]
        nflats = chain_samples.shape[0] * chain_walkers
            
        resampled    = np.full((nsample, npars_3G), np.nan, dtype=float)
        resampled_1G = np.full((nsample, len(names_1G)), np.nan, dtype=float)

        pbar = tqdm(total=nsample) if pbar_resample else None
        timei = time.time()
        
        yy   = self.y
        leny = len(yy)
        noise_rows = residual_noise_rows(residuals_orig, nsample, leny)
        
        V1,S1,B1 = self.df.loc[0,['V1','S1','B1']]
        
        fitB1 = self.fit_params_1G['B1']
                
        if fitB1 == 'BB':
            def iter_resample(y_w_noise, guess_XG):
                BB = makefit_bg_linefree_njit(xx, y_w_noise, e_y, V1, S1, B1)
                gmodel_resample_1G.BB = BB
                gmodel_resample_XG.BB = BB
                res_1G = minimize(gmodel_resample_1G.log_prob_resample, guess_1G, args=(y_w_noise, BB), method=method_minimize)
                res_XG = minimize(gmodel_resample_XG.log_prob_resample, guess_XG, args=(y_w_noise, BB), method=method_minimize)
                return res_1G, res_XG

        elif fitB1 == 'B1':
            self.fit_params_1G['B1']='free'
            try:
                gmodel_1G_Bfree,_ = self.prep_1GFIT()
            finally:
                self.fit_params_1G['B1']='B1'
            guess_1G_Bfree = self.df.loc[0, ['F1','V1','S1','B1']].to_numpy(dtype=np.float64)
            def iter_resample(y_w_noise, guess_XG):
                B1 = fit_resampled_baseline(gmodel_1G_Bfree, guess_1G_Bfree,
                                           y_w_noise, method_minimize)
                gmodel_resample_1G.BB = B1
                gmodel_resample_XG.BB = B1
                res_1G       = minimize(gmodel_resample_1G.log_prob_resample, guess_1G, args=(y_w_noise, B1), method=method_minimize)
                res_XG       = minimize(gmodel_resample_XG.log_prob_resample, guess_XG, args=(y_w_noise, B1), method=method_minimize)
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
        width_history = []

        for j in range(nsample):
            noise = next(noise_rows)
            while True:
                y_w_noise = yy + noise
                rdm_index = np.random.choice(nflats)
                
                guess_XG = chain_samples[rdm_index // chain_walkers, rdm_index % chain_walkers]
                
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
                
                if j>=1000:
                    width_history.append(compute_widths(resampled, j, npars_3G))
                    converged = check_converged_resampling(width_history)
                    if converged:
                        resampled    = resampled[   :j,:]
                        resampled_1G = resampled_1G[:j,:]
                        BBs          = BBs[:j]
                        funs         = funs[:j]
                        self.df_params['Nsample'] = j
                        break
                
                timei = timef

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

        # Release the replay batch/view before allocating physical samples and
        # constructing potentially large diagnostic figures.
        noise_rows.close()
        del noise_rows, chain_samples
            
        
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
        self.checkpoint_plot_state(self.gmodel)
        self.make_atlas(cleanup=True)
        
        return
