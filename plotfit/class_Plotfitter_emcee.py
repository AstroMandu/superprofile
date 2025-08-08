import datetime
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Literal

import emcee
import numpy as np
import pandas as pd
import pylab as plt
from scipy.optimize import minimize
from scipy.stats import skew, percentileofscore, kurtosis
from tqdm import tqdm

from .class_Plotfitter_gmodel import Gmodel
from .class_Plotfitter_plotter import Plotter
from .subroutines_Plotfitter import do_Ftest, gauss, gaussian_area, calc_red_chisq_1G, calc_red_chisq_2G, get_mode, sort_outliers, get_asymmetry_residuals


class Plotfit:
    
    def __init__(self, df_stacked, dict_disp, name_cube=None, path_plot=None, path_temp=None, plot_autocorr=False, longest_length=None, vdisp_low_intrinsic=4.):
        
        df_stacked = df_stacked.loc[df_stacked['y']!=0].reset_index(drop=True)
        
        self.df_stacked = df_stacked
        
        self.x   = np.asarray(self.df_stacked['x'], dtype=np.float64)
        self.y   = np.asarray(self.df_stacked['y'], dtype=np.float64)
        self.e_y = np.asarray(self.df_stacked['e_y'], dtype=np.float64)
        
        # self.e_y = np.full_like(self.x, 1.)
        
        self.longest_length = 0 if longest_length is None else longest_length

        if(name_cube is None):
            self.name_cube = 'Name'
        else:
            self.name_cube = name_cube
        
        df = pd.DataFrame({'Name':[self.name_cube]})
        columns = []
        columns = np.append(columns, 'SNR1')
        columns = np.append(columns, 'SNR2')
        columns = np.append(columns, 'N1')
        columns = np.append(columns, 'N2')
        
        columns = np.append(columns, 'F-test')
        columns = np.append(columns, 'F-crit')
        
        # columns = np.append(columns, 'Skew_A21')
        # columns = np.append(columns, 'Skew_A22')
        # columns = np.append(columns, 'Skew_S21')
        # columns = np.append(columns, 'Skew_S22')

        columns = np.append(columns,    'B1')
        columns = np.append(columns,    'A1')
        columns = np.append(columns,    'V1')
        columns = np.append(columns,    'S1')
        
        columns = np.append(columns,    'B2')
        columns = np.append(columns, 'e-_B2')
        columns = np.append(columns, 'e+_B2')
        columns = np.append(columns,    'A21')
        columns = np.append(columns, 'e-_A21')
        columns = np.append(columns, 'e+_A21')
        columns = np.append(columns,    'A22')
        columns = np.append(columns, 'e-_A22')
        columns = np.append(columns, 'e+_A22')
        columns = np.append(columns,    'V21')
        columns = np.append(columns, 'e-_V21')
        columns = np.append(columns, 'e+_V21')
        columns = np.append(columns,    'V22')
        columns = np.append(columns, 'e-_V22')
        columns = np.append(columns, 'e+_V22')
        columns = np.append(columns,    'S21')
        columns = np.append(columns, 'e-_S21')
        columns = np.append(columns, 'e+_S21')
        columns = np.append(columns,    'S22')
        columns = np.append(columns, 'e-_S22')
        columns = np.append(columns, 'e+_S22')
        
        for column in columns: df[column] = np.nan
        self.df = df
        
        self.df_resampled = pd.DataFrame({'Name':[self.name_cube]})
        self.df_params    = pd.DataFrame({'Name':[self.name_cube]})
        
        self.df_params['Reliable'] = 'N'
        self.df_params['SNR2']    = np.nan
        self.df_params[  'sn']    = np.nan
        self.df_params[  'sb']    = np.nan
        self.df_params[  'An']    = np.nan
        self.df_params[  'Ab']    = np.nan
        self.df_params[  'At']    = np.nan
        self.df_params['e_sn']    = np.nan
        self.df_params['e_sb']    = np.nan
        self.df_params['e_An']    = np.nan
        self.df_params['e_Ab']    = np.nan
        self.df_params['e_At']    = np.nan
        self.df_params[  'sn/sb'] = np.nan
        self.df_params[  'An/At'] = np.nan
        self.df_params['e_sn/sb'] = np.nan
        self.df_params['e_An/At'] = np.nan
        
        self.dict_disp = dict_disp
        
        self.list_disp   = np.array([dict_disp[i]['disp'] for i in dict_disp.keys()])
        # self.e_list_disp = np.array([dict_disp[i]['e_disp'] for i in dict_disp.keys()])
        
        if(path_plot is not None): path_plot = Path(path_plot)
        if(path_temp is not None): path_temp = Path(path_temp)
        self.path_plot = path_plot
        self.path_temp = path_temp
        self.plot_autocorr = plot_autocorr
        
        self.dict_params = {
            'A21':'free', # free only
            'A22':'free', # free only
            'V21':'fix', # fix will make V21=V1
            'V22':'fix',  # fix will make V22=V21
            'S21':'free', # free only
            'S22':'free', # free only
            'B2' :'fix'  # fix will make B1=B2
        }
        
        self.GFIT1_success = False
        self.GFIT2_success = False
        self.resample_success = False
        self.timei = np.nan
        self.stat = 'Start'
        
        self.testlength: int = 1000
        self.truth_from_resampling: bool = False
        self.slope_tau_1G: float = 50
        self.slope_tau_2G: float = 100
        
        self.maxiter_1G = 50000
        self.maxiter_2G = 100000
        
        self.gmodel = None
        self.sampler   = None
        self.resampled = None
        
        self.statistics: Literal['mean','median','mode'] = 'median'
        
        if self.statistics=='median': self.skew_thres = 0.7
        if self.statistics=='mode':   self.skew_thres = 0.9
        
        self.vdisp_low_intrinsic = vdisp_low_intrinsic
        
    def writestat(self, message):
        f = open(self.path_temp/f'stat.{self.name_cube}.txt', 'w')
        f.write(f'{self.name_cube:>{self.longest_length}} {message}')
        f.close()
        # with open(self.path_temp/f'Plotfit_{self.name_cube:>{self.longest_length}}_stat.txt', 'w') as f:
        return
        
    def removestat(self):
        path_stat = self.path_temp/f'stat.{self.name_cube}.txt'
        if os.path.exists(path_stat):
            os.remove(path_stat)
        return

    def print_diagnose_params(self, gmodel:Gmodel, params:np.ndarray[float]) -> None:
            
        if 'A21' in gmodel.names_param:
            
            df_diag = pd.DataFrame()
            df_diag['Index'] = ['orig','mapd','fini']
            
            # params        = gmodel.array_to_dict_guess(params)
            params_mapped = gmodel.array_to_dict_guess(gmodel.map_params(params).flatten())
            params        = gmodel.array_to_dict_guess(params)
            df_diag['S1'] = [gmodel.S1,None,None]
            for i, label in enumerate(gmodel.names_param):
                df_diag[label] = [params[label],params_mapped[label],True if (params_mapped[label]<1 and params_mapped[label]>0) else False]
            if np.any(df_diag.loc[df_diag['Index']=='fini']==False):
                pprint(gmodel.dict_bound)
                print(df_diag.to_string())
        
            if 'B2' in gmodel.names_param:
                df_diag = pd.DataFrame()
                df_diag['cond']    = ['ampl']
                # df_diag['A21+A22'] = np.sum([params_mapped['A21'],params_mapped['A22']])
                df_diag['fini']    = False if gmodel.test_2G_ampl(params['A21'],params['A22'],params['B2'])==False else True
                if np.any(df_diag['fini']==False):
                    print(df_diag.to_string())
            
            df_diag = pd.DataFrame()
            df_diag['cond']       = ['disp1']
            df_diag['S1']         = gmodel.S1
            df_diag['S21']        = params['S21']
            df_diag['S22']        = params['S22']
            df_diag['S22-S21']    = df_diag['S22']-df_diag['S21']
            df_diag['delta_disp'] = gmodel.delta_disp
            df_diag['fini'] = False if gmodel.test_2G_disp_order(params['S21'],params['S22'])==False else True
            if np.any(df_diag['fini']==False):
                print(df_diag.to_string())
            
            # df_diag = pd.DataFrame()
            # df_diag['cond'] = ['disp2']
            # df_diag['S1']  = self.df.loc[0,'S1']
            # df_diag['S21']  = params['S21']
            # df_diag['S22']  = params['S22']
            # df_diag['S1-S21'] = df_diag['S1'] - df_diag['S21']
            # df_diag['S22-S1'] = df_diag['S22'] - df_diag['S1']
            # df_diag['fini']    = True if gmodel.log_prior_G2_disp2([params['S21'],params['S22']])==0.0 else False            
            # print(df_diag.to_string())
            
        else:
            df_diag = pd.DataFrame()
            df_diag['Index'] = ['orig','mapd','fini']
            
            params        = gmodel.array_to_dict_guess(params)
            params_mapped = gmodel.array_to_dict_guess(gmodel.map_params(params).flatten())
            for i, label in enumerate(gmodel.names_param):
                df_diag[label] = [params[label],params_mapped[label],True if (params_mapped[label]<1 and params_mapped[label]>0) else False]
            if np.any(df_diag.loc[df_diag['Index']=='fini']==False):
                pprint(gmodel.dict_bound)
                print(df_diag.to_string())
            
        return 

    
    def makeplot_autocorr(self, 
                          gmodel: Gmodel, 
                          converged: bool) -> None:
        
        self.savename_autocorr = self.path_plot / 'Plotfit_autocorr_{}.png'.format(self.name_cube)
        
        if(self.path_plot is None): return
            
        xs = self.chekstep
        
        fig_ac, ax_ac = plt.subplots()
        ax_ac.plot(xs, xs/self.slope_tau, '--k', label=r'$N$'+f'={self.slope_tau:.0f}'+r'$\tau$')
        ax_ac.plot(xs, self.autocorr_mean, label='mean')
        ax_ac.plot(xs, self.autocorr_max,  label='max ({})'.format(gmodel.names_param[self.autocorr_argmax]))
        
        ax_ac.set_xlabel('Steps')
        ax_ac.set_ylabel(r'$\hat{\tau}$')
        ax_ac.set_title(self.name_cube)
        ax_ac.legend()
        fig_ac.savefig(self.savename_autocorr,dpi=100)
        
        if(converged):
            ax_ac.axvline(int(2 * np.max(self.old_tau)), color='tab:red')
            if(os.path.exists(self.savename_autocorr)):
                os.remove(self.savename_autocorr)
                
        plt.close(fig_ac)
        del fig_ac
        return
    
    def makecalc_ETA_emcee(self, index):
        
        if self.timei==np.nan: 
            self.timei = time.time()
            return
        
        y1, y2 = self.autocorr_max[index-1], self.autocorr_max[index]
        x1, x2 = (index-1)*self.testlength, index*self.testlength
        
        slope = (y2-y1)/self.testlength
        if slope>1./self.slope_tau: 
            x_to_go = self.maxiter-x2
        else:
            x_intersect = np.min([(slope*x1-y1)/(slope - 1./self.slope_tau),self.maxiter])
            x_to_go = x_intersect-x2
        if(x_to_go<0): return 'Soon'
        eta = x_to_go / (self.testlength/(time.time()-self.timei))
        self.timei = time.time()
        
        if np.isnan(eta): return 'nan'
        return datetime.timedelta(seconds=int(eta))
    
    
    def check_converged(self, 
                        sampler:emcee.EnsembleSampler, 
                        gmodel:Gmodel, 
                        generate_plot:bool=False) -> bool:
        
        iter = sampler.iteration
        tau  = sampler.get_autocorr_time(tol=0)
        if(np.any(np.isnan(tau))):
            last_sample = sampler.get_chain(flat=True)[-1,:]
            print(self.name_cube, tau, gmodel.array_to_dict_guess(last_sample))
            self.print_diagnose_params(gmodel, last_sample)
            
        if(iter>self.maxiter): converged=True
        else:
            converged  = np.all(tau*self.slope_tau < iter)
            converged &= np.all(np.abs(self.old_tau - tau) / tau < 0.01)
        
        self.old_tau = tau
        index = int(iter/self.testlength)
        self.chekstep[index] = iter
        self.autocorr_mean[index] = np.mean(tau)
        self.autocorr_argmax = np.argmax(tau)
        self.autocorr_max[index] = tau[self.autocorr_argmax]
        eta = self.makecalc_ETA_emcee(index)
        self.writestat(f'{self.stat} {iter} {eta}')
        if(generate_plot):
            self.makeplot_autocorr(gmodel, converged)
            
        if converged: self.timei = np.nan
        return converged
    
    def makefit_1G_minimize(self, xx, yy, e_y, guess=None, return_gmodel=False):
        
        if self.dict_params['V21']==0: names_param = np.array(['A1','S1','B1'])
        else:                          names_param = np.array(['A1','V1','S1','B1'])
        
        dict_bound = {
            'A1':np.array([0,1.5])*yy.max(),
            'V1':np.array([-3*self.chansep,3*self.chansep]),
            'S1':np.array([self.dispmin,self.dispmax]),
            'B1':np.array([-3.0,3.0])*yy.max()
        }
        
        gmodel = Gmodel(xx,yy,e_y,names_param,dict_bound)
        gmodel.log_prob = gmodel.log_prob_1G
        
        if guess is None:
            guess = np.array([self.ymax,0.0,10.0,0.0])
        
        res = minimize(gmodel.log_prob_guess, guess, method='Nelder-Mead')
        
        if return_gmodel: return gmodel.array_to_dict_guess(res.x), gmodel
        return gmodel.array_to_dict_guess(res.x)
        
    def limit_range(self, multiplier_disp:float=10) -> None:
        
        res_1G = self.makefit_1G_minimize(self.x, self.y, self.e_y)
        
        V1,S1 = res_1G['V1'], res_1G['S1']
        xi = np.array(np.max([V1 - S1*multiplier_disp, self.xmin]))
        xf = np.array(np.min([V1 + S1*multiplier_disp, self.xmax]))
        
        df_limited = self.df_stacked.loc[self.df_stacked['x'].between(xi,xf)]
        self.x   = np.array(df_limited['x'])
        self.y   = np.array(df_limited['y']) #+ np.random.normal(0,self.df['N1'][0],len(self.x))
        self.e_y = np.array(df_limited['e_y'])
        
        self.xmin, self.xmax = np.array(np.nanpercentile(self.x, [0,100]))
        self.ymin, self.ymax = np.array(np.nanpercentile(self.y, [0,100]))
        self.chansep = np.array(np.abs(np.mean(np.diff(self.x))))
        self.bandwidh = self.xmax-self.xmin
        
        return
        
    def fill_df_emcee(self, sampler, names, burnin, thin=1):
        if thin==0: thin==1
        
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_probs = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)    # shape (n_steps * n_walkers)
        max_prob_index = np.argmax(log_probs)
        
        if self.statistics=='median':
            for i, label in enumerate(names):
                percentiles = np.percentile(flat_samples[:,i], [16,50,84])
                self.df[label]       = flat_samples[max_prob_index,i]# percentiles[1]
                self.df['e-_'+label] = percentiles[1]-percentiles[0]
                self.df['e+_'+label] = percentiles[2]-percentiles[1]
        if self.statistics=='mode':
            for i, label in enumerate(names):
                mode = get_mode(flat_samples[:,i])
                percentile_mode = percentileofscore(flat_samples[:,i], mode)
                percentile_mode = np.max([0.0, percentile_mode]); percentile_mode = np.min([100.0, percentile_mode])
                percentiles = np.percentile(flat_samples[:,i], [np.max([percentile_mode-34,0]), percentile_mode, np.min([percentile_mode+34,100])])
                self.df[label]       = percentiles[1]
                self.df['e-_'+label] = percentiles[1]-percentiles[0]
                self.df['e+_'+label] = percentiles[2]-percentiles[1]
                
        
        if ('A1' in names) and ('V1' not in names):
            self.df['V1'] = 0.0
            
        if ('A21' in names) and ('V21' not in names): self.df['V21']=self.df['V1'][0]
        if ('A21' in names) and ('V22' not in names): self.df['V22']=self.df['V21'][0]
        if ('A21' in names) and ('B2'  not in names): self.df['B2'] =self.df['B1'][0]
        
        if 'A21' in names:
            self.flat_samples = flat_samples
        
    def makefit_1G(self,
                   xx:np.ndarray[float]=None, 
                   yy:np.ndarray[float]=None, 
                   e_y:np.ndarray[float]=None,
                   guess:bool=None):
        
        self.maxiter   = self.maxiter_1G
        self.slope_tau = self.slope_tau_1G
        
        if xx  is None:  xx=self.x
        if yy  is None:  yy=self.y
        if e_y is None: e_y=self.e_y
        
        res_1G,gmodel = self.makefit_1G_minimize(xx,yy,e_y, guess=guess, return_gmodel=True)
        guess = [res_1G[param] for param in gmodel.names_param]
        self.gmodel = gmodel

        self.stat = '1GFIT'
        # pos = guess + 1e-5*np.array(np.random.randn(int(len(guess)*2), len(guess)))#self.G*3)
        # nwalkers, ndim = pos.shape
        
        nwalkers = int(len(guess)*2)
        ndim = len(guess)

        lower_bounds = np.array([gmodel.dict_bound[param][0] for param in gmodel.names_param])
        upper_bounds = np.array([gmodel.dict_bound[param][1] for param in gmodel.names_param])

        pos = guess + 1e-5 * np.random.randn(nwalkers, ndim)
        pos = np.clip(pos, lower_bounds, upper_bounds)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gmodel.log_prob_1G,
                                        # parameter_names=gmodel.names_param,
                                        moves=[
                                            (emcee.moves.DEMove(), 0.8),
                                            (emcee.moves.DESnookerMove(), 0.2),
                                            ]
                                        )
        arrlen = int(self.maxiter/self.testlength)
        self.autocorr_mean = np.full(arrlen+1, np.nan)
        self.autocorr_max  = np.full(arrlen+1, np.nan)
        self.chekstep      = np.full(arrlen+1, np.nan)
        self.old_tau = np.inf
        for sample in sampler.sample(pos, iterations=self.maxiter):#, progress=True):
            iteration = sampler.iteration
            if(iteration%self.testlength): continue
            if(self.check_converged(sampler, gmodel, generate_plot=False)): break
            
        # Fill df with results
        self.burnin = int(2   * np.max(self.old_tau))
        self.thin   = int(0.5 * np.min(self.old_tau))
        if(self.thin==0): self.thin = 1
        self.fill_df_emcee(sampler, gmodel.names_param, self.burnin, self.thin)
        self.writestat(f'{self.stat} - Done')
        
        self.df_params[  'S1'] = self.df[  'S1'][0]
                
        A1,V1,S1,B1 = self.df.loc[0,['A1','V1','S1','B1']]
        
        residual = yy - (gauss(xx, A1,V1,S1)+B1)
        self.residual_1GFIT = residual
        
        N1   = np.std(residual)
        SNR1 = A1/N1
        
        self.df['SNR1'] = SNR1
        self.df['N1']   = N1
        
        self.GFIT1_success = True
        
        
        
        return

    def make_guess(self, 
                   dict_1Gfit:dict=None, 
                   dict_bound:dict=None) -> tuple[np.ndarray[float], np.ndarray[str]]:
            
        dict_1Gfit = dict_1Gfit or {
            'N1': self.df['N1'][0],
            'B1': self.df['B1'][0],
            'A1': self.df['A1'][0],
            'V1': self.df['V1'][0],
            'S1': self.df['S1'][0]
        }
            
        dict_guess = {
            'A21':dict_1Gfit['A1']*0.45,
            'A22':dict_1Gfit['A1']*0.45,
            'V21':dict_1Gfit['V1'],
            'V22':dict_1Gfit['V1'],
            'S21':dict_1Gfit['S1'],
            # 'S21':self.dispmin+0.1,
            # 'S21':dict_1Gfit['S1']*0.5+0.1,
            'S22':dict_1Gfit['S1']+0.1,
            'B2' :dict_1Gfit['B1']
        }
        
        if dict_guess['S21']>dict_guess['S22']: dict_guess['S21'] = (dict_1Gfit['S1']+self.dispmin)/2.
        
        names_param = np.array([key for key, status in self.dict_params.items() if status == 'free'])
        guess       = np.array([dict_guess[param] for param in names_param])
            
        return guess, names_param
    
    def makefit_2G(self, maxiter:int=100000) -> None:
        
        if self.GFIT1_success==False:
            self.df_params['Reliable'] = 'N'
            return
        
        self.stat = '2GFIT'
        self.maxiter = self.maxiter_2G
        self.slope_tau = self.slope_tau_2G
        
        S1 = self.df['S1'][0]
        B1 = self.df['B1'][0]
        N1 = self.df['N1'][0]

        dict_bound = {
            'A21':np.array([-0.001*self.ymax, self.ymax*1.5]), #upbnd_Amp]),#np.max([A1,self.ymax])]),
            'A22':np.array([-0.001*self.ymax, self.ymax*1.5]), #upbnd_Amp]),#np.max([A1,self.ymax])]),
            'V21':np.array([-5*S1,5*S1]),
            'V22':np.array([-5*S1,5*S1]),
            # 'S21':np.array([self.dispmin, self.dispmax]),
            # 'S22':np.array([self.dispmin, self.dispmax]),
            
            'S21':np.array([self.dispmin, self.dispmax]),
            'S22':np.array([self.dispmin, self.dispmax]),
            
            'B2' :np.array([B1-N1,B1+N1])
            # 'B2' :np.array([-self.ymax*0.5, self.ymax*0.5])
        }
        
        if np.diff(dict_bound['S21'])<0:
            print("[Plotfit] {} 2GFIT is a no go; 1GFIT near at bound".format(self.name_cube))
            self.df_params['Reliable'] = 'nearbound'
            return
        
        guess,names_param = self.make_guess(dict_bound=dict_bound)
        
        gmodel = Gmodel(self.x, self.y, self.e_y, names_param, dict_bound, self.df)
        gmodel.log_prob = gmodel.log_prob_2G
        
        self.guess_2gfit  = guess
        self.gmodel       = gmodel
        self.params_2gfit = names_param
        self.dict_bound   = dict_bound.copy()

        if(np.any(np.isfinite(gmodel.log_prior_2G_diagnose(guess))==False)):
            pprint(dict_bound)
            self.print_diagnose_params(gmodel,guess)
            self.make_atlas()
            
            raise ValueError("[Plotfit] {}'s initial guess violate the priors".format(self.name_cube))
        
        # guess = minimize(gmodel.log_prob_guess, guess, method='Nelder-Mead', tol=1e-10).x
        # if(np.any(np.isfinite(gmodel.log_prior(dict(zip(gmodel.names_param, guess))))==False)):
        #     guess,_ = self.make_guess()
        
        # nwalkers = int(len(guess)*2)
        # pos = guess + 1e-7*np.array(np.random.randn(nwalkers, len(guess)))#self.G*3)
        # nwalkers, ndim = pos.shape
        
        nwalkers = int(len(guess)*2)
        ndim     = len(guess)

        lower_bounds = np.array([gmodel.dict_bound[param][0] for param in gmodel.names_param])
        upper_bounds = np.array([gmodel.dict_bound[param][1] for param in gmodel.names_param])

        pos = guess + 1e-5 * np.random.randn(nwalkers, ndim)
        pos = np.clip(pos, lower_bounds, upper_bounds)
        
        # for i in range(nwalkers):
        #     if(np.any(np.isfinite(gmodel.log_prior_2G(guess)==False))):
        #         print(self.name_cube,self.print_diagnose_params(gmodel, pos[i,:]))
        #         guess,names_param = self.make_guess(dict_bound=dict_bound)
        #         self.guess_2gfit  = guess
        #         pos = guess + 1e-5*np.array(np.random.randn(int(len(guess)*2), len(guess)))#self.G*3)
        #         break
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        gmodel.log_prob,
                                        # parameter_names=gmodel.names_param,
                                        moves=[
                                            (emcee.moves.DEMove(), 0.8),
                                            (emcee.moves.DESnookerMove(), 0.2),
                                            ]
                                        )
        self.sampler = sampler
        
        arrlen = int(self.maxiter/self.testlength)
        self.autocorr_mean = np.full(arrlen+1, np.nan)
        self.autocorr_max  = np.full(arrlen+1, np.nan)
        self.chekstep      = np.full(arrlen+1, np.nan)
        self.old_tau = np.inf   
        
        for sample in sampler.sample(pos, iterations=self.maxiter):#, progress=True):
            iteration = sampler.iteration
            if(iteration%self.testlength): continue
            if(self.check_converged(sampler, gmodel, generate_plot=self.plot_autocorr)): break
            
        if self.plot_autocorr:
            if os.path.exists(self.savename_autocorr): os.remove(self.savename_autocorr)
            
        # Fill df with results
        self.burnin = int(2   * np.max(self.old_tau))    
        self.thin   = 0#int(0.5 * np.min(self.old_tau))
        if(self.thin==0): self.thin = 1
        self.fill_df_emcee(sampler, gmodel.names_param, self.burnin, self.thin)
        
        # Calculate residual
        A21,A22,V21,V22,S21,S22,B2 = self.df.loc[0,['A21','A22','V21','V22','S21','S22','B2']]
        model_narw = gauss(self.x, A21,V21,S21)+B2/2
        model_brod = gauss(self.x, A22,V22,S22)+B2/2
        model_totl = model_narw+model_brod
        
        residual = self.y - model_totl
        self.residual_2GFIT = residual
        
        self.df['SNR2'] = (A21+A22)/np.nanstd(residual)
        self.df['N2']   = np.nanstd(residual)
        
        F, crit = do_Ftest(gmodel, self.df)
        self.df['F-test'] = F
        self.df['F-crit'] = crit
        
        A21,A22,S21,S22 = self.df.loc[0,['A21','A22','S21','S22']]
        An,Ab = gaussian_area(A21,S21), gaussian_area(A22,S22)
        
        self.df_params['SNR2'] = self.df['SNR2'][0]
                
        self.df_params[  'sn'] = S21
        self.df_params[  'sb'] = S22
        self.df_params[  'An'] = An
        self.df_params[  'Ab'] = Ab
        self.df_params[  'At'] = An+Ab
        
        self.df_params[  'sn/sb'] = self.df_params['sn'][0]/self.df_params['sb'][0]
        self.df_params[  'An/At'] = self.df_params['An'][0]/self.df_params['At'][0]
        
        self.df_params['Asym'] = get_asymmetry_residuals(self.y,residual)
        
        # print(self.name_cube, self.df_params['Asym'][0])
        
        self.sampler    = sampler
        
        self.GFIT2_success = True
        self.writestat(f'{self.stat} - Done')
        
        return
    
    def resample(self, nsample:int=1499, pbar_resample:bool=False) -> None:
        
        SNR1, SNR2 = self.df.loc[0,['SNR1','SNR2']]
        
        if np.isfinite(SNR1)==False or np.isfinite(SNR2)==False:
            return
        
        self.nsample = nsample
        self.df_params['SNR2'] = SNR2
        self.stat = 'Resample'
        
        names_param = np.array(self.params_2gfit)
        guess = np.zeros(len(names_param))
        for i, label in enumerate(names_param):
            guess[i] = self.df[label][0]
        resampled = np.full((self.nsample,len(guess)),np.nan)
        
        S1s = np.full(self.nsample, np.nan)
        
        guess_1G = self.df.loc[0,['A1','V1','S1','B1']]
        N1 = self.df['N1'][0]
                
        if pbar_resample:
            pbar = tqdm(total=nsample)
            
        trueiter = 0
        timei = time.time()
        
        for j in range(nsample):
            
            while True:
                # self.x_w_noise = xx     + np.random.normal(0,self.chansep/2,len(xx))
                # y_w_noise = self.y + np.random.normal(0,self.df['N2'][0],len(self.y))
                y_w_noise = self.y + np.random.choice(self.residual_2GFIT, len(self.y), replace=True)
                ymax = y_w_noise.max()
                
                gmodel = self.gmodel
                gmodel.y = y_w_noise
                
                res_1G    = self.makefit_1G_minimize(self.x, y_w_noise, self.e_y, guess=guess_1G)
                A1, V1, S1, B1 = (res_1G[key] for key in ['A1','V1','S1','B1'])
                
                gmodel.update_bound('A21',np.array([-0.001, 1.5], dtype=np.float64)*ymax)
                gmodel.update_bound('A22',np.array([-0.001, 1.5], dtype=np.float64)*ymax)
                # if self.dict_params['B2']=='free':
                #     gmodel.update_bound('B2',np.array([B1-N1,B1+N1],dtype=np.float64))
                
                res = minimize(gmodel.log_prob_guess, guess, method='Nelder-Mead', tol=1e-8)
                # print(gmodel.names_param)
                # print(gmodel.return_bounds_list())
                # res = minimize(gmodel.log_prob_guess, guess, bounds=gmodel.return_bounds_list(), method='Powell', tol=1e-8)
                trueiter+=1
                if np.isfinite(res.fun):
                    S1s[j] = S1
                    resampled[j,:] = res.x
                    break
                
            if j%100==0 and j!=0:
                timef = time.time()
                eta = datetime.timedelta(seconds=int((nsample-j)/(100/(timef-timei))))
                self.writestat(f'Resample {j} {eta}')
                self.resampled = resampled[:j,:]
                timei = timef
            
            if pbar_resample: pbar.update()
        
        self.resampled = resampled    
        # df_temp = pd.DataFrame()
        # for i,param in enumerate(names_param):
        #     df_temp[param] = self.resampled[:,i]
        # print(df_temp.to_string())
            
        for i, label in enumerate(names_param):
            data_label = resampled[:,i]
            self.df_resampled[label]      = np.nanmean(data_label)
            self.df_resampled['e_'+label] = np.nanstd(data_label)
            
        def fillstat(label, data):
            self.df_params[label]      = np.nanmean(data)
            self.df_params['e_'+label] = np.nanstd(data)
            
        sns = resampled[:,np.argwhere(self.params_2gfit=='S21')[0]]
        sbs = resampled[:,np.argwhere(self.params_2gfit=='S22')[0]]
        Ans = gaussian_area(resampled[:,np.argwhere(self.params_2gfit=='A21')[0]], sns)
        Abs = gaussian_area(resampled[:,np.argwhere(self.params_2gfit=='A22')[0]], sbs)
        Ats = Ans+Abs
        
        if self.truth_from_resampling:
            print(f'[Plotfit {self.name_cube}] truth_from_resampling is True; filling df with resampled values')
            for param in gmodel.names_param:
                self.df[param] = np.median(sort_outliers(resampled[:,np.argwhere(self.params_2gfit==param)[0]]))
        
        A21,A22,S21,S22 = self.df.loc[0,['A21','A22','S21','S22']]
        An,Ab = gaussian_area(A21,S21), gaussian_area(A22,S22)
        
        self.df_params[  'sn'] = S21
        self.df_params[  'sb'] = S22
        self.df_params[  'An'] = An
        self.df_params[  'Ab'] = Ab
        self.df_params[  'At'] = An+Ab
        
        self.df_params['e_S1'] = np.nanstd(S1s)
        
        self.df_params['e_sn'] = np.nanstd(sns)
        self.df_params['e_sb'] = np.nanstd(sbs)
        self.df_params['e_An'] = np.nanstd(Ans)
        self.df_params['e_Ab'] = np.nanstd(Abs)
        self.df_params['e_At'] = np.nanstd(Ats)
        
        if self.truth_from_resampling:
            snsbs = sns/sbs
            snsbs = snsbs[snsbs<0.9]
            self.df_params['sn/sb']      = np.median(sort_outliers(snsbs))
            self.df_params['An/At']      = np.median(sort_outliers(Ans/Ats))
            self.df_params['log(sb-sn)'] = np.nanmedian(sort_outliers(np.log10(sbs-sns)))
        else:
            self.df_params[  'sn/sb']    = self.df_params['sn'][0]/self.df_params['sb'][0]
            self.df_params[  'An/At']    = self.df_params['An'][0]/self.df_params['At'][0]
            self.df_params['log(sb-sn)'] = np.log10(self.df_params['sb'][0]-self.df_params['sn'][0])
            
        self.df_params['e_sn/sb'] = np.nanstd(snsbs)
        self.df_params['e_An/At'] = np.nanstd(Ans/Ats)
        self.df_params['e_log(sb-sn)'] = np.nanstd(np.log10(sbs-sns))
        
        # fillstat('sn', sns)
        # fillstat('sb', sbs)
        # fillstat('An', Ans)
        # fillstat('Ab', Abs)
        # fillstat('At', Ats)
        # fillstat('sn/sb', sns/sbs)
        # fillstat('An/At', Ans/Ats)
        
        self.resampled = resampled
        self.resample_success = True
        self.df_params['Reliable'] = 'Y'
        
        return

    
    
    def check_config(self) -> None:
        
        self.dict_params['A21'] = 'free'
        self.dict_params['A22'] = 'free'
        self.dict_params['S21'] = 'free'
        self.dict_params['S22'] = 'free'
        
        if self.dict_params['V21']==0:
            self.dict_params['V21'] = '0'
            
        if self.dict_params['V21'] not in ['0','free','fix']:
            raise TypeError('[Plotfit] V21 has to be one of either "0", "free" or "fix".')
        
        if self.dict_params['V22'] not in ['0','free','fix']:
            raise TypeError('[Plotfit] V22 has to be one of either "0", "free" or "fix".')
            
        if self.dict_params['B2'] not in ['free','fix']:
            raise TypeError('[Plotfit] B2 has to be one of either "free" or "fix".')
        
        if self.dict_params['V21']=='0':
            if self.dict_params['V22']=='free':
                print('[Plotfit] Setting V22=fix; V22 free is not possible when V21==0')
                self.dict_params['V22'] = 'fix'
        
        if self.dict_params['V21']=='fix':
            if self.dict_params['V22']=='free':
                print('[Plotfit] Setting V22=fix; V22 free is not possible when V21==fix')
                self.dict_params['V22'] = 'fix'
    
    def check_stacked(self):
        
        # if(len(self.x)<10): 
        #     self.df_params['Reliable']='N'
        #     print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Array length is too small to make a meaningful fit (less than 5).")
        #     self.df_params['Reliable']='0stacked'
        #     return False
        if np.all(self.y==0) or np.all(np.isfinite(self.y)==False):
            self.df_params['Reliable']='0stacked'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Nothing seem to be stacked.")
            return False
        # if len(self.list_disp)<2:
        #     self.df_params['Reliable']='0stacked'
        #     print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; (Almost) Nothing seem to be stacked.")
        #     return False
        
        self.xmin, self.xmax = np.nanpercentile(self.x, [0,100])
        self.ymin, self.ymax = np.nanpercentile(self.y, [0,100])
        self.chansep = np.abs(np.mean(np.diff(self.x)))
        self.bandwidh = self.xmax-self.xmin

        return True
    
    def check_list_disp(self):
        
        # list_disp, inidces = sort_outliers(self.list_disp, return_index=True, only_higher=True)
        # e_list_disp = self.e_list_disp[inidces]
        
        list_disp = self.list_disp
        
        # self.min_list_disp = np.min(list_disp)
        # self.max_list_disp = np.max(list_disp)
        
        # self.med_list_disp = np.median(list_disp)
        # self.mean_list_disp = np.mean(list_disp)
        
        # self.dispmin = np.min([np.max([np.min(self.list_disps) - self.chansep/2., self.chansep/2.]), np.min(self.list_disps)])
        # self.dispmin = np.min(self.list_disps) # - self.chansep/4.
        # self.dispmax = np.max([100,self.max_list_disps]) #np.max(self.list_disps) + self.chansep/4.
        #====
        # argwhere = np.argwhere(self.kde_y>0.0001)[0][0]
        # self.dispmin = np.max([self.kde_x[argwhere],0])
        # # self.dispmin = self.min_list_disps-self.chansep/4.
        
        # argwhere = np.argwhere(self.kde_y>0.0001)[-1][0]
        # # self.dispmax = np.max([100,self.max_list_disps])#self.kde_x[argwhere]
        # # self.dispmax = np.max(self.list_disps)#+self.chansep/4
        # self.dispmax = self.kde_x[argwhere]
        
        # argmin = np.argmin(self.list_disp)
        # argmax = np.argmax(self.list_disp)
        # self.dispmin = self.list_disp[argmin] - self.e_list_disp[argmin]
        # self.dispmax = self.list_disp[argmax] + self.e_list_disp[argmax]
        
        # self.dispmin = np.min(list_disp-e_list_disp)
        # self.dispmax = np.max(list_disp+e_list_disp)
        
        # self.dispmin = np.min(list_disp)
        # self.dispmax = np.max(list_disp)
        
        self.dispmin = self.chansep/2.355 * 3
        self.dispmax = self.bandwidh/2.355
        
        # self.dispmin = self.chansep/2.355  # np.sqrt(self.vdisp_low_intrinsic**2 + (self.chansep/2.355)**2)
        # self.dispmax = np.max(self.list_disp) #self.bandwidh/2.355
        
        # if self.dispmin<self.chansep/4.: self.dispmin = self.chansep/4.
        
        # print(self.name_cube, self.list_disp[argmin], self.e_list_disp[argmin])
        #====
        
        # self.percentile_maxdisp = 90
        
        # while self.percentile_maxdisp<=100:
        #     if self.percentile_maxdisp<100:
        #         self.max_list_disps = np.percentile(self.list_disps, self.percentile_maxdisp)
        #     else:
        #         self.max_list_disps = self.dispmax
        #     if self.max_list_disps>self.min_list_disps: break
        #     self.percentile_maxdisp+=10
            
        if self.dispmax-self.dispmin<2:#self.chansep/2:
            self.df_params['Reliable']='disprng'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Range of stacked dispersion is narrow")
            return False
        
        # self.percentile_maxdisp = percentile_maxdisp
        
        return True
    

    
    
    def evaluate_1GFIT(self):
        A1, V1, S1, B1, SNR1 = self.df.loc[0,['A1','V1','S1','B1','SNR1']]
        
        # if SNR1<10: 
        #     print("[Plotfit] {} 2GFIT is a no go; 1GFIT SNR<10".format(self.name_cube))
        #     self.df_params['Reliable'] = 'lowSNR1'
        #     self.GFIT1_success = False
        #     return
        
        if S1*2.3*2>self.bandwidh: #FWHM*3<bandwidth
            print("[Plotfit] {} 2GFIT is a no go; Bandwidth is too narrow to make a meaningful fit".format(self.name_cube))
            self.df_params['Reliable'] = 'W_S1>BW'
            self.GFIT1_success = False
            return
        
        # if np.abs(self.dict_bound_1G['S1'][1]-self.df.loc[0,'S1'])<1e-1: 
        #     print("[Plotfit] {} 2GFIT is a no go; 1GFIT near at bound".format(self.name_cube))
        #     self.df_params.loc[0,'Reliable'] = 'nearbound'
        #     self.GFIT1_success = False
        #     return
        
        # if np.abs(self.df.loc[0,'S1']-self.dict_bound_1G['S1'][0])<1e-1: 
        #     print("[Plotfit] {} 2GFIT is a no go; 1GFIT near at bound".format(self.name_cube))
        #     self.df_params.loc[0,'Reliable'] = 'nearbound'
        #     self.GFIT1_success = False
        #     return
    
    def evaluate_2GFIT(self):
        
        A21,A22,N2,SNR2 = self.df.loc[0,['A21','A22','N2','SNR2']]
        # print(self.name_cube, A21,A22,N2)
        
        if SNR2<15:
            self.df_params['Reliable'] = 'lowSNR2'
            # print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling is a no go; SNR2<15.")
            self.GFIT2_success = False
            return
        
        if A21<3*N2:
            self.df_params['Reliable'] = 'lowA21'
            # print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling is a no go; A21 meaningless.")
            self.GFIT2_success = False
            return
        if A22<3*N2:
            self.df_params['Reliable'] = 'lowA22'
            # print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling is a no go; A22 meaningless.")
            self.GFIT2_success = False
            return

        
        # F, crit = self.df.loc[0,['F-test','F-crit']]
        # if F<crit:
        #     print(f"[Plotfit] {self.name_cube} 2GFIT is a no go; Failed F-test: F={F:.1f}, ({crit:.1f}).")
        #     self.df_params['Reliable'] = 'F-fail'
        #     self.GFIT2_success = False
        #     return
        
        # if calc_red_chisq_1G(self.gmodel, self.df)<calc_red_chisq_2G(self.gmodel, self.df):
        #     print(f"[Plotfit] {self.name_cube} χ_1G < χ_2G")
        #     self.df_params['Reliable'] = 'chi1G<chi2G'
        #     self.GFIT2_success = False
        #     return 
        
        for param in ['A21','A22','S21','S22']:
            argwhere_param = np.argwhere(self.params_2gfit==param)[0]
            params = self.flat_samples[:,argwhere_param]
            
            # perc_param = np.percentile(params, (2.5,97.5))
            # params = params[(params>perc_param[0])&(params<perc_param[1])]
            
            params = sort_outliers(params)
            
            skew_param = skew(params)
            kurt_param = kurtosis(params)
            self.df[f'Skew_{param}'] = skew_param
            self.df[f'Kurt_{param}'] = kurt_param

        skew_A21 = self.df['Skew_A21'][0]
        skew_A22 = self.df['Skew_A22'][0]
        skew_S21 = self.df['Skew_S21'][0]
        skew_S22 = self.df['Skew_S22'][0]
        
        kurt_A21 = self.df['Kurt_A21'][0]
        kurt_A22 = self.df['Kurt_A22'][0]
        kurt_S21 = self.df['Kurt_S21'][0]
        kurt_S22 = self.df['Kurt_S22'][0]
        
        if skew_S21>self.skew_thres and not skew_S22<-self.skew_thres:
            self.df_params['Reliable'] = 'S21_at_edge'
            self.GFIT2_success = False
            return
            
        if not skew_S21>self.skew_thres and skew_S22<-self.skew_thres:
            self.df_params['Reliable'] = 'S22_at_edge'
            self.GFIT2_success = False
            return
            
        if skew_S21<-self.skew_thres and skew_S22>self.skew_thres:
            self.df_params['Reliable'] = 'S21~S22'
            self.GFIT2_success = False
            return
        
        if np.abs(skew_S21)>self.skew_thres and np.abs(skew_S22)>self.skew_thres:
            self.df_params['Reliable'] = 'disps_at_edge'
            self.GFIT2_success = False
            return
        
        # kurt_thres = -0.6
        # if kurt_A21<kurt_thres or kurt_A22<kurt_thres or kurt_S21<kurt_thres or kurt_S22<kurt_thres:
        # if kurt_S21<kurt_thres:
        #     self.df_params['Reliable'] = 'Dist.flat'
        #     self.GFIT2_success = False
        #     return
            
        # if np.abs(skew_A21)>.6 or np.abs(skew_A22)>.6:
        #     self.df_params['Reliable'] = 'ampls_at_edge'
        #     self.GFIT2_success = False
        #     return
            
        return
    
    def make_atlas(self):
        plotter = Plotter(self.path_plot, self.name_cube, self.suffix,
                          self.list_disp, self.df, self.df_params,
                          self.gmodel, self.sampler, self.resampled)
        plotter.makeplot_atlas()
        return
    
    
    def run(self, 
            suffix:str='', 
            nsample_resample:int=1499, 
            pbar_resample:  bool=False) -> None:
        
        self.suffix = suffix
        self.nsample_resample = nsample_resample
        
        self.check_config()
        if self.check_stacked()==False:   return
        if self.check_list_disp()==False: return
        
        self.limit_range(multiplier_disp=15)
        self.makefit_1G(self.x, self.y, self.e_y)
        self.evaluate_1GFIT()
        self.make_atlas()
        
        # if self.GFIT1_success:
        #     self.makefit_2G()
        #     self.make_atlas()
            
        #     # if self.GFIT2_success:
        #     #     self.redo()
        #     #     self.make_atlas()
            
        #     if self.GFIT2_success:
        #         self.evaluate_2GFIT()
        #         self.make_atlas()
            
        #     if self.GFIT2_success:           
        #         self.resample(nsample=nsample_resample, pbar_resample=pbar_resample)
        #         # self.test_skew()
        #         self.make_atlas()
            
        if self.GFIT1_success:
            self.makefit_2G()
            self.make_atlas()
            
            self.resample(nsample=nsample_resample, pbar_resample=pbar_resample)
            self.evaluate_2GFIT()
            self.make_atlas()
            
        self.removestat()
        