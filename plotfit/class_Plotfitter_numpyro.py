import time
from pprint import pprint
import os
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
import pylab as plt
from chainconsumer import Chain, ChainConsumer
from routines_baygaudpi import gauss
from scipy.optimize import minimize
from .class_Plotfitter_gmodel import Gmodel
from typing import Literal
from scipy.stats import skew
import seaborn as sns
import datetime
from scipy.stats import gaussian_kde
from scipy.stats import f
from PIL import Image
import numpyro
from numpyro import distributions as dist, infer
import jax
import jax.numpy as jnp
import numpyro

class Plotfit:
    
    def __init__(self, df_stacked, dict_disp, name_cube=None, path_plot=None, path_temp=None, plot_autocorr=False, longest_length=None, maxtaskperchild=1):
        
        self.maxtaskperchild = maxtaskperchild
        numpyro.set_host_device_count(maxtaskperchild)
        
        df_stacked = df_stacked.loc[df_stacked['y']!=0].reset_index(drop=True)
        
        self.df_stacked = df_stacked
        
        self.x   = np.array(self.df_stacked['x'])
        self.y   = np.array(self.df_stacked['y'])
        self.e_y = np.array(self.df_stacked['e_y'])
        
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
        
        columns = np.append(columns, 'Skew_A21')
        columns = np.append(columns, 'Skew_A22')
        columns = np.append(columns, 'Skew_S21')
        columns = np.append(columns, 'Skew_S22')

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
        
        if(len(self.x)<5): 
            self.df_params['Reliable']='N'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Array length is too small to make a meaningful fit (less than 5).")
            return
        
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
        
        self.xmin, self.xmax = np.nanpercentile(self.x, [0,100])
        self.ymin, self.ymax = np.nanpercentile(self.y, [0,100])
        self.chansep = np.abs(np.mean(np.diff(self.x)))
        self.bandwidh = self.xmax-self.xmin
        
        self.dict_disp = dict_disp
        self.list_disp   = np.array([dict_disp[i]['disp'] for i in dict_disp.keys()])
        self.e_list_disp = np.array([dict_disp[i]['e_disp'] for i in dict_disp.keys()])
        
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
        
        self.GFIT1_succecs = False
        self.GFIT2_success = False
        self.resample_success = False
        self.multiplier_S1 = 0.9
        self.timei = np.nan
        self.stat = 'Start'
        
        self.testlength = 1000
        self.truth_from_resampling = False
        self.slope_tau_1G = 50
        self.slope_tau_2G = 50
        
        self.maxiter_1G = 50000
        self.maxiter_2G = 50000
        
    def writestat(self, message):
        f = open(self.path_temp/f'stat.{self.name_cube}.txt', 'w')
        f.write(f'{self.name_cube:>{self.longest_length}}.{message}')
        f.close()
        # with open(self.path_temp/f'Plotfit_{self.name_cube:>{self.longest_length}}_stat.txt', 'w') as f:
        return
        
    def removestat(self):
        path_stat = self.path_temp/f'stat.{self.name_cube}.txt'
        if os.path.exists(path_stat):
            os.remove(path_stat)
        return

    def makefit_1G(self,
                   xx:np.ndarray[float]=None, 
                   yy:np.ndarray[float]=None, 
                   e_y:np.ndarray[float]=None,
                   save_on_df:bool=True, mcmc:bool=False, guess:bool=None,
                   mode=Literal['emcee','resample']):

        self.maxiter = self.maxiter_1G
        
        if xx  is None:  xx=self.x
        if yy  is None:  yy=self.y
        if e_y is None: e_y=self.e_y
        
        def model_1G(xx,e_y,yy=None):
            A1 = numpyro.sample('A1', dist.Uniform(0,1.5*self.ymax))
            V1 = numpyro.sample('V1', dist.Uniform(-3*self.chansep,3*self.chansep))
            S1 = numpyro.sample('S1', dist.Uniform(self.min_list_disp, self.max_list_disp))
            B1 = numpyro.sample('B1', dist.Uniform(-0.3*self.ymax,0.3*self.ymax))
            
            with numpyro.plate('data', len(xx)):
                numpyro.sample('yy', dist.Normal(A1*jnp.exp(-0.5*((xx-V1)/S1)**2)+B1, e_y), obs=yy)
        
        kernel = infer.NUTS(model_1G)
        mcmc   = infer.MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=4, progress_bar=True)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, xx, e_y, yy=yy)
        samples = mcmc.get_samples()
        self.stat = '1GFIT'
        for i, label in enumerate(['A1','V1','S1','B1']):
            # percentiles = np.array(np.percentile(flat_samples[:,i], [16,50,84]))
            percentiles = np.percentile(samples[label], [16,50,84])
            self.df[label]       = percentiles[1]
            self.df['e-_'+label] = percentiles[1]-percentiles[0]
            self.df['e+_'+label] = percentiles[2]-percentiles[1]
        # if 'V1' not in gmodel.name_params: self.df['V1'] = 0.0
        
        self.writestat(f'{self.stat}.-.Done')
        
        A1,V1,S1,B1 = self.df.loc[0,['A1','V1','S1','B1']]
        N1   = np.std(yy - (gauss(xx, A1,V1,S1)+B1))
        SNR1 = A1/N1
        
        if(save_on_df):
            self.df['SNR1'] = SNR1
            self.df['N1']   = N1

        self.GFIT1_success = True
        
        
        # if 'V1' not in gmodel.name_params:
        #     return {'A1':res.x[0],'V1':0.0,     'S1':res.x[1],'B1':res.x[2],'N1':N1,'SNR1':SNR1,'res':res}
        # else:
        #     return {'A1':res.x[0],'V1':res.x[1],'S1':res.x[2],'B1':res.x[3],'N1':N1,'SNR1':SNR1,'res':res}
    
    def limit_range(self, multiplier_disp:float=10) -> None:
        
        if not np.isfinite(self.df['SNR1'].item()): 
            self.makefit_1G(self.x, self.y, self.e_y)
        
        V1,S1 = self.df.loc[0,['V1','S1']]
        xi = np.array(np.max([V1 - S1*multiplier_disp, self.xmin]))
        xf = np.array(np.min([V1 + S1*multiplier_disp, self.xmax]))
        
        df_limited = self.df_stacked.loc[self.df_stacked['x'].between(xi,xf)]
        self.x   = np.array(df_limited['x'])
        self.y   = np.array(df_limited['y']) #+ np.random.normal(0,self.df['N1'].item(),len(self.x))
        self.e_y = np.array(df_limited['e_y'])
        
        self.xmin, self.xmax = np.array(np.nanpercentile(self.x, [0,100]))
        self.ymin, self.ymax = np.array(np.nanpercentile(self.y, [0,100]))
        self.chansep = np.array(np.abs(np.mean(np.diff(self.x))))
        self.bandwidh = self.xmax-self.xmin
        
        return
        
    def calc_red_chisq_1G(self):
        A1, V1, S1, B1 = self.df.loc[0, ['A1', 'V1', 'S1', 'B1']]
        res_1G   = self.y - (gauss(self.x, A1,V1,S1)+B1)
        chisq_1G = np.sum((res_1G/self.e_y)**2) 
        dof_1G   = len(self.y)-4
        
        return chisq_1G/dof_1G
    
    def calc_red_chisq_2G(self):
        A21, A22, V21, V22, S21, S22, B2 = self.df.loc[0, ['A21','A22','V21','V22','S21','S22','B2']]
        
        res_2G   = self.y - (gauss(self.x, A21,V21,S21)+gauss(self.x, A22,V22,S22)+B2)
        chisq_2G = np.sum((res_2G/self.e_y)**2) 
        dof_2G   = len(self.y) - len(self.params_2gfit) 
        
        return chisq_2G/dof_2G
    
    def do_Ftest(self, significance=0.05):
        
        # calculate 1Gres
        A1, V1, S1, B1 = self.df.loc[0, ['A1', 'V1', 'S1', 'B1']]
        res_1G   = self.y - (gauss(self.x, A1,V1,S1)+B1)
        chisq_1G = np.sum((res_1G/self.e_y)**2) 
        dof_1G   = len(self.y)-4

        A21, A22, V21, V22, S21, S22, B2 = self.df.loc[0, ['A21','A22','V21','V22','S21','S22','B2']]
        
        res_2G   = self.y - (gauss(self.x, A21,V21,S21)+gauss(self.x, A22,V22,S22)+B2)
        chisq_2G = np.sum((res_2G/self.e_y)**2) 
        dof_2G   = len(self.y) - len(self.params_2gfit) 
        
        F = (chisq_1G - chisq_2G)/(dof_1G - dof_2G)/(chisq_2G/dof_2G)
        critical_value = f.ppf(1 - significance, dof_1G-dof_2G, dof_2G)
        
        return F, critical_value        
    
    def makefit_2G(self, maxiter:int=100000) -> None:
        
        if self.GFIT1_success==False:
            self.df_params['Reliable'] = 'N'
            return
        
        A1, V1, S1, B1, SNR1 = self.df.loc[0,['A1','V1','S1','B1','SNR1']]
        
        if SNR1<10: 
            print("[Plotfit] {} 2GFIT is a no go; 1GFIT SNR<10".format(self.name_cube))
            self.df_params['Reliable'] = 'lowSNR1'
            return
        
        if S1*2.3*2>self.bandwidh: #FWHM*3<bandwidth
            print("[Plotfit] {} 2GFIT is a no go; Bandwidth is too narrow to make a meaningful fit".format(self.name_cube))
            self.df_params['Reliable'] = 'velrange'
            return
        
        # if np.abs(self.dict_bound_1G['S1'][1]-self.df.loc[0,'S1'])<1e-1: 
        #     print("[Plotfit] {} 2GFIT is a no go; 1GFIT near at bound".format(self.name_cube))
        #     self.df_params.loc[0,'Reliable'] = 'nearbound'
        #     return
        
        # if np.abs(self.df.loc[0,'S1']-self.dict_bound_1G['S1'][0])<1e-1: 
        #     print("[Plotfit] {} 2GFIT is a no go; 1GFIT near at bound".format(self.name_cube))
        #     self.df_params.loc[0,'Reliable'] = 'nearbound'
        #     return
        
        self.stat = '2GFIT'
        
        def model_2G(xx, e_y, yy=None):
            
            A21 = numpyro.sample('A21', dist.Uniform(0,1.5*self.ymax))
            A22 = numpyro.sample('A22', dist.Uniform(0,1.5*self.ymax))
            
            S21 = numpyro.sample('S21', dist.Uniform(self.dispmin,self.df['S1'].item()))
            S22 = numpyro.sample('S22', dist.Uniform(self.df['S1'].item(),self.dispmax))
            
            B2 = numpyro.sample('B2', dist.Uniform(-0.3*self.ymax,0.3*self.ymax))
            
            G1 = A21*jnp.exp(-0.5*((xx-self.df['V1'].item())/S21)**2)
            G2 = A22*jnp.exp(-0.5*((xx-self.df['V1'].item())/S22)**2)
            
            with numpyro.plate("data", len(xx)):
                numpyro.sample("yy", dist.Normal(G1+G2+B2, e_y), obs=yy)

        kernel = infer.NUTS(model_2G)
        mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=50000, num_chains=5, progress_bar=True)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, self.x, self.e_y, yy=self.y)
        
        self.mcmc = mcmc
        
        self.params_2gfit = ['A21','A22','S21','S22','B2']
        
        samples = mcmc.get_samples()
        for i, label in enumerate(['A21','A22','S21','S22','B2']):
            percentiles = np.percentile(samples[label], [16,50,84])
            self.df[label]       = percentiles[1]
            self.df['e-_'+label] = percentiles[1]-percentiles[0]
            self.df['e+_'+label] = percentiles[2]-percentiles[1]
        
        # if('V21' not in name_params): self.df['V21']=V1
        # if('V22' not in name_params): self.df['V22']=self.df['V21'].item()
        # if('B2'  not in name_params): self.df['B2'] =B1
        
        self.df['V21']=V1
        self.df['V22']=self.df['V21'].item()
        # self.df['B2'] =B1
            
        # Calculate residual
        A21,A22,V21,V22,S21,S22,B2 = self.df.loc[0,['A21','A22','V21','V22','S21','S22','B2']]
        model_narw = gauss(self.x, A21,V21,S21)+B2
        model_brod = gauss(self.x, A22,V22,S22)+B2
        model_totl = model_narw+model_brod-B2
        
        residual = self.y - model_totl
        
        self.df['SNR2'] = (A21+A22)/np.nanstd(residual)
        self.df['N2']   = np.nanstd(residual)
        
        F, crit = self.do_Ftest()
        self.df['F-test'] = F
        self.df['F-crit'] = crit
        
        def gaussian_area(amp, sigma):
            return amp*sigma*np.sqrt(2*np.pi)
        
        A21,A22,S21,S22 = self.df.loc[0,['A21','A22','S21','S22']]
        An,Ab = gaussian_area(A21,S21), gaussian_area(A22,S22)
        self.df_params[  'sn'] = S21
        self.df_params[  'sb'] = S22
        self.df_params[  'An'] = An
        self.df_params[  'Ab'] = Ab
        self.df_params[  'At'] = An+Ab
        
        self.df_params[  'sn/sb'] = self.df_params['sn'].item()/self.df_params['sb'].item()
        self.df_params[  'An/At'] = self.df_params['An'].item()/self.df_params['At'].item()
        
        self.GFIT2_success = True

        self.writestat(f'{self.stat}.-.Done')
        
        return
        
    def resample(self, nsample:int=1499, pbar_resample:bool=False) -> None:
        
        def minmax(iterable):
            it = iter(iterable)
            try:
                min_val = max_val = next(it)  # Initialize with the first value
            except StopIteration:
                raise ValueError("minmax() arg is an empty sequence")
            for x in it:
                if x < min_val:
                    min_val = x
                elif x > max_val:
                    max_val = x
            return min_val, max_val
        
        SNR1, SNR2 = self.df.loc[0,['SNR1','SNR2']]
        
        self.nsample = nsample
        self.df_params['SNR2'] = SNR2
        self.stat = 'Resample'
        
        if np.isfinite(SNR1)==False or np.isfinite(SNR2)==False:
            return
        
        if self.GFIT2_success==False:
            return
        
        # if self.df_params.loc[0,'Reliable']=='nearbound':
        #     print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling won't be made; Params near at bounds.")
        #     return
        # # print(self.df[['e-_S21','e+_S21','e-_S22','e+_S22']].to_string())
        
        # if self.df.loc[0,'e-_S21']<1e-3 and self.df.loc[0,'e+_S22']<1e-3:
        #     self.df_params.loc[0,'Reliable'] = 'nearbound'
        #     print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling won't be made; S21 and S22 near at bounds.")
        #     return
        
        F, crit = self.do_Ftest()
        if F<crit:
            print(f"[Plotfit] {self.name_cube} 2GFIT is a no go; Failed F-test: F={F:.1f}, ({crit:.1f}).")
            self.df_params['Reliable'] = 'F-fail'
            return
        
        if self.calc_red_chisq_1G()<self.calc_red_chisq_2G():
            print(f"[Plotfit] {self.name_cube} χ_1G < χ_2G")
            self.df_params['Reliable'] = 'chi1G<chi2G'
            return 
        
        A21,A22,S21,S22,N2 = self.df.loc[0,['A21','A22','S21','S22','N2']]
        if N2==0: return
        
        name_params = np.array(self.params_2gfit)
        guess = np.zeros(len(name_params))
        for i, label in enumerate(name_params):
            guess[i] = self.df[label].item()
        resampled = np.full((self.nsample,len(guess)),np.nan)
        
        if pbar_resample:
            from tqdm import tqdm
            pbar = tqdm(total=nsample)
            
        trueiter = 0
        timei = time.time()
        
        A21, A22, V21, V22, S21, S22, B2 = self.df.loc[0, ['A21','A22','V21','V22','S21','S22','B2']]
        res_2G   = self.y - (gauss(self.x, A21,V21,S21)+gauss(self.x, A22,V22,S22)+B2)
        chisq_2G = np.sum((res_2G/self.e_y)**2) 
        
        for j in range(nsample):
            
            while True:
                # self.x_w_noise = xx     + np.random.normal(0,self.chansep/2,len(xx))
                y_w_noise = self.y + np.random.normal(0,N2,len(self.y))
                ymin,ymax = minmax(y_w_noise)
                # ymax = y_w_noise.max()
                # ymin_abs = np.abs(ymin)
                
                gmodel = self.gmodel_2GFIT
                gmodel.y = y_w_noise
                
                res_1G    = self.makefit_1G(yy=y_w_noise, save_on_df=False, guess=self.guess_1G, mode='resample')
                A1, V1, S1, B1 = (res_1G[key] for key in ['A1','V1','S1','B1'])
                        
                model_G1 = gauss(self.x_masked, A1, V1, S1)+B1
                resid_G1 = y_w_noise - model_G1
                
                maxexcess = np.nanmax(np.abs(resid_G1))
                upbnd_Amp = A1+maxexcess*2

                gmodel.dict_bound['A21'] = np.array([-0.001*ymax, self.ymax*1.5])#upbnd_Amp])
                gmodel.dict_bound['A22'] = np.array(gmodel.dict_bound['A21'])
                # gmodel.dict_bound['S21'] = np.float32([self.min_list_disps,bnd_disp])
                # gmodel.dict_bound['S22'] = np.float32([bnd_disp,self.max_list_disps])
                # gmodel.dict_bound['S21'] = np.float32([0,200])
                # gmodel.dict_bound['S22'] = np.float32([0,200])
                gmodel.dict_bound['S21'] = np.array([self.dispmin,S1])
                gmodel.dict_bound['S22'] = np.array([S1,self.dispmax])
                
                if self.dict_params['B2']=='free':
                    # gmodel.dict_bound['B2'] = np.array([-1*ymax,ymax])*0.3
                    gmodel.dict_bound['B2']=np.array([B1-maxexcess, B1+maxexcess])
                    gmodel.dict_bound['divB2'] = np.array(1/np.diff(gmodel.dict_bound['B2'])[0])
                # gmodel.dict_bound['S22'] = np.array([res_1G['S1'],res_1G['S1']*2-self.min_list_disps])
                
                for key in ['A21','A22','S21','S22']: gmodel.dict_bound['div'+key] = np.array(1/np.diff(gmodel.dict_bound[key])[0])
                
                argwhere_S21 = np.argwhere(name_params=='S21').item()
                if guess[argwhere_S21]<gmodel.dict_bound['S21'][0] or guess[argwhere_S21]>gmodel.dict_bound['S21'][1]:
                    guess[argwhere_S21] = np.array(np.mean(gmodel.dict_bound['S21']))
                # argwhere_B2 = np.argwhere(name_params=='B2').item()
                # guess[argwhere_B2] = 0    
                
                # guess, name_params = self.make_guess(dict_1Gfit=res_1G, dict_bound=gmodel.dict_bound)
                res = minimize(gmodel.log_prob_guess, guess, method='Nelder-Mead', tol=1e-100)#, tol=chisq_2G*0.01)  
                trueiter+=1
                if np.isfinite(res.fun):
                    
                    xs = np.linspace(self.xmin, self.xmax, 1000)
                    A21 = res.x[np.argwhere(self.params_2gfit=='A21').item()]
                    A22 = res.x[np.argwhere(self.params_2gfit=='A22').item()]
                    S21 = res.x[np.argwhere(self.params_2gfit=='S21').item()]
                    S22 = res.x[np.argwhere(self.params_2gfit=='S22').item()]
                    if 'B2' in self.params_2gfit:
                        B2 = res.x[np.argwhere(self.params_2gfit=='B2').item()]
                    else:
                        B2 = self.df.loc[0,'B2']
                    V21,V22 = self.df.loc[0,['V21','V22']]
                    
                    residual = self.y - (gauss(self.x,A21,V21,S21)+gauss(self.x,A22,V22,S22)+B2)
                    
                    N2 = np.std(residual)
                    
                    # if A21>3*N2 and A22>3*N2:
                    if True:
                    
                        resampled[j,:] = res.x
                        
                        # model_narw = gauss(xs, A21,V21,S21)+B2/2
                        # model_brod = gauss(xs, A22,V22,S22)+B2/2
                        # model_totl = model_narw+model_brod
                        
                        # savename = self.path_plot / "emcee_2GFIT_resampled_{:03}_{}_{}.png".format(j,self.suffix,self.name_cube)
                        
                        # fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw={'height_ratios':[5,1]},
                        #                         figsize=(8,5.5),
                        #                         tight_layout=True)
                        

                        
                        # ax = axs[0][0]
                        # ax.errorbar(self.x, y_w_noise, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
                        # ax.plot(xs, model_narw, color='tab:blue'  ,       label=r'$\sigma$={:.1f}'.format(S21))
                        # ax.plot(xs, model_brod, color='tab:orange',       label=r'$\sigma$={:.1f}'.format(S22))
                        # ax.plot(xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
                        # ax.legend()
                        
                        # ax = axs[1][0]        
                        # ax.plot(self.x, residual, color='tab:blue')
                            
                        # for ax in axs.flatten(): ax.axhline(0, color='gray', alpha=0.5)
                        # axs[0][0].set_ylabel(r'$\mathrm{Jy}$')
                        # axs[1][0].set_xlabel(r'$\mathrm{km \ s^{-1}}$')
                        
                        # fig.suptitle(f'{self.name_cube:>{self.longest_length}} {self.suffix}')
                        # fig.savefig(savename)
                        # plt.close(fig)
                    
                        break
                
                # if (trueiter-j)%200==0: 
                #     print(self.name_cube, '{}/{}'.format(j,nsample), trueiter)
                #     self.print_diagnose_params(gmodel, res.x)
                    
                #     savename = self.path_plot / "emcee_1GFIT{}_{}.png".format(self.suffix,self.name_cube)
                    
                #     fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw={'height_ratios':[5,1]},
                #                             figsize=(8,5.5),
                #                             tight_layout=True)
                    
                #     xs = np.linspace(self.xmin, self.xmax, 1000)
                #     residual = y_w_noise - (gauss(self.x,A1,V1,S1)+B1)
                    
                #     ax = axs[0][0]
                #     ax.errorbar(self.x, y_w_noise, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
                #     ax.plot(xs, gauss(xs, A1,V1,S1)+B1, color='tab:blue', label=r"S/N={:.1f}\n".format(res_1G['SNR1'])+r"$\sigma$={:.1f}".format(S1))
                #     ax.legend(loc='upper right')
                    
                #     ax = axs[1][0]        
                #     ax.plot(self.x, residual, color='tab:blue')
                        
                #     for ax in axs.flatten(): ax.axhline(0, color='gray', alpha=0.5)
                #     axs[0][0].set_ylabel(r'$\mathrm{Jy}$')
                #     axs[1][0].set_xlabel(r'$\mathrm{km \ s^{-1}}$')
                    
                #     fig.suptitle(f'{self.name_cube:>{self.longest_length}} {self.suffix}')
                #     fig.savefig(savename)
                #     plt.close(fig)

            if j%100==0 and j!=0:
                timef = time.time()
                eta = datetime.timedelta(seconds=(nsample-j)/(100/(timef-timei)))
                self.writestat(f'Resample.{j}.{eta}')
                self.resampled = resampled[:j,:]
                self.makeplot_walks(transparent=False)
                timei = timef
            
            # print(res)
            
            # print(gmodel.array_to_dict_guess(res.x))
            # if res.fun==np.inf:
            #     print(self.name_cube)
            #     print(self.df[['e-_S21','e+_S21','e-_S22','e+_S22']].to_string())
            #     self.print_diagnose_params_G2(gmodel, res.x)
            # print(res)
            
            # if np.isfinite(res.fun):
            #     resampled[j,:] = res.x
            
            if pbar_resample: pbar.update()
            
            # del gmodel
        
        # perc = np.nanpercentile(resampled, (0.1,99.9), axis=1)
        # mask = (resampled >= perc[0, :, None]) & (resampled <= perc[1, :, None])
        # resampled = np.where(mask, resampled, np.nan)
        
        try:
            os.remove(self.path_plot / "emcee_1GFIT{}_{}.png".format(self.suffix,self.name_cube))
        except:
            pass
            
        for i, label in enumerate(name_params):
            data_label = resampled[:,i]
            self.df_resampled[label]      = np.nanmean(data_label)
            self.df_resampled['e_'+label] = np.nanstd(data_label)
            
        def fillstat(label, data):
            self.df_params[label]      = np.nanmean(data)
            self.df_params['e_'+label] = np.nanstd(data)
            
        def gaussian_area(amp, sigma):
            return amp*sigma*np.sqrt(2*np.pi)
            
        sns = resampled[:,np.argwhere(self.params_2gfit=='S21').item()]
        sbs = resampled[:,np.argwhere(self.params_2gfit=='S22').item()]
        Ans = gaussian_area(resampled[:,np.argwhere(self.params_2gfit=='A21').item()], sns)
        Abs = gaussian_area(resampled[:,np.argwhere(self.params_2gfit=='A21').item()], sbs)
        Ats = Ans+Abs
        
        if self.truth_from_resampling:
            for param in gmodel.name_params:
                print(param)
                self.df[param] = np.mean(resampled[:,np.argwhere(self.params_2gfit==param).item()])
        
        A21,A22,S21,S22 = self.df.loc[0,['A21','A22','S21','S22']]
        An,Ab = gaussian_area(A21,S21), gaussian_area(A22,S22)
        self.df_params[  'sn'] = S21
        self.df_params[  'sb'] = S22
        self.df_params[  'An'] = An
        self.df_params[  'Ab'] = Ab
        self.df_params[  'At'] = An+Ab
        
        self.df_params['e_sn'] = np.nanstd(sns)
        self.df_params['e_sb'] = np.nanstd(sbs)
        self.df_params['e_An'] = np.nanstd(Ans)
        self.df_params['e_Ab'] = np.nanstd(Abs)
        self.df_params['e_At'] = np.nanstd(Ats)
            
        self.df_params[  'sn/sb'] = self.df_params['sn'].item()/self.df_params['sb'].item()
        self.df_params[  'An/At'] = self.df_params['An'].item()/self.df_params['At'].item()
        self.df_params['e_sn/sb'] = np.nanstd(sns/sbs)
        self.df_params['e_An/At'] = np.nanstd(Ans/Ats)
        
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
        
        
    def makeplot_corner_mcmc(self, savefig:bool=True) -> plt.Figure:
        
        savename_corner = self.path_plot / "emcee_corner{}_{}.png".format(self.suffix, self.name_cube)
        
        chain = Chain.from_numpyro(self.mcmc, "numpyro")
        consumer = ChainConsumer().add_chain(chain)
        
        try:
            fig = consumer.plotter.plot()
        except:
            return
        
        if savefig:
            fig.savefig(savename_corner, transparent=True)
            self.savename_corner_emcee = savename_corner
        
        return fig
        
    def makeplot_corner_resample(self, savefig:bool=True) -> plt.Figure:
        
        # if self.df_params['Reliable'].item()!='Y':
        #     return
        
        if ~np.isfinite(self.df_params['e_sn'].item()): return
        
        def gaussian_area(amp, sigma):
            return amp*sigma*np.sqrt(2*np.pi)
        
        savename_corner = self.path_plot / "resampled_corner{}_{}.png".format(self.suffix, self.name_cube)
        df = pd.DataFrame()
        
        for i, label in enumerate(self.params_2gfit):
            # print(self.df)
            # print(self.df_params)
            df[label] = self.resampled[:,i]
        
        # df['sn'] = df['S21']
        # df['sb'] = df['S22']
        # df['An'] = gaussian_area(df['A21'], df['S21'])
        # df['Ab'] = gaussian_area(df['A22'], df['S22'])
        # df['At'] = df['An']+df['Ab']
        # df['sn/sb'] = df['S21']/df['S22']
        # df['An/At'] = df['An']/df['At']
                        
        chain = Chain(samples=df, name='Resampled')
        consumer = ChainConsumer().add_chain(chain)
        
        # chain = Chain.from_emcee(self.sampler, self.params_2gfit, discard=self.burnin, thin=self.thin, name='Sampled')
        # consumer.add_chain(chain)
        
        from chainconsumer import Truth
        
        dict_truth = {}
        for param in self.params_2gfit:
            dict_truth[param] = self.df[param].item()
        consumer.add_truth(Truth(location=dict_truth, name='emcee', color='tab:blue'))
        for i, param in enumerate(self.params_2gfit):
            dict_truth[param] = np.nanmedian(self.resampled[:,i])
        consumer.add_truth(Truth(location=dict_truth, name='resampled', color='tab:orange'))
        
        try:
            fig = consumer.plotter.plot()
        except:
            return
        
        if savefig:
            fig.savefig(savename_corner, transparent=True)
            self.savename_corner_resample = savename_corner
        
        return fig
    
    def makeplot_walks(self, savefig:bool=True, transparent:bool=True) -> plt.Figure:
        
        if self.df_params['Reliable'].item()!='Y':
            return
        
        savename_walks = self.path_plot / "Plotfit_walks{}_{}.png".format(self.suffix, self.name_cube)
        
        # chain = Chain.from_emcee(self.sampler, self.params_2gfit, discard=self.burnin, thin=self.thin, name='Sampled')
        # consumer = ChainConsumer().add_chain(chain)
        
        df = pd.DataFrame()
        for i, label in enumerate(self.params_2gfit):
            # print(self.df)
            # print(self.df_params)
            df[label] = self.resampled[:,i]
        
        # df['sn'] = df['S21']
        # df['sb'] = df['S22']
        # df['An'] = gaussian_area(df['A21'], df['S21'])
        # df['Ab'] = gaussian_area(df['A22'], df['S22'])
        # df['At'] = df['An']+df['Ab']
        # df['sn/sb'] = df['S21']/df['S22']
        # df['An/At'] = df['An']/df['At']
                        
        chain = Chain(samples=df, name='Resampled')
        consumer = ChainConsumer().add_chain(chain)

        try:
            fig = consumer.plotter.plot_walks(convolve=100, plot_weights=False)
        except:
            return
        
        if savefig:
            fig.savefig(savename_walks, transparent=transparent)
            self.savename_walks = savename_walks
        
        return fig

    def makeplot_1GFIT(self, ax_1GFIT, ax_1Gres):
        xs = np.linspace(self.xmin, self.xmax, 1000)
        
        SNR1, A1, V1, S1, B1 = self.df.loc[0,['SNR1','A1','V1','S1','B1']]
        residual = self.y - (gauss(self.x,A1,V1,S1)+B1)
        self.res_1G = residual
        
        ax = ax_1GFIT
        ax.set_title('1G', fontsize=20)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        ax.plot(xs, gauss(xs, A1,V1,S1)+B1, color='tab:blue', label=r"$\sigma$={:.1f}".format(S1))
        ax.legend(title='S/N={:.0f}'.format(SNR1), loc='upper right')
        ax.set_ylabel(r'$\mathrm{Jy}$')
        
        ax_1Gres.axhline(0, color='gray', alpha=0.5)
        ax_1Gres.scatter(self.x, residual, s=3, color='tab:blue')
        ax_1Gres.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
        
        chisq = np.sum((residual/self.e_y)**2)
        dof   = len(self.y)-4
        chisq_red = chisq/dof
        ax_1Gres.text(0.99,0.99, r'$\chi^2_\mathrm{red}$='+f'{chisq_red:.1f}', va='top',ha='right', transform=ax_1Gres.transAxes)
        
        noise = np.std(residual)
        ax_1Gres.text(0.99,0.01, r'RMS='+f'{noise*1000:.2f} mJy', va='bottom',ha='right', transform=ax_1Gres.transAxes)
        
        self.chisq_1G = chisq
        
    def makeplot_2GFIT(self, ax_2GFIT, ax_2Gres):
        xs = np.linspace(self.xmin, self.xmax, 1000)
        ax_2GFIT.xaxis.set_tick_params(labelbottom=False)
        ax_2GFIT.yaxis.set_tick_params(  labelleft=False)
        ax_2Gres.yaxis.set_tick_params(  labelleft=False)
        
        #plot G2
        ax = ax_2GFIT
        ax.set_title('2G', fontsize=20)
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        SNR2 = self.df['SNR2'].item()
        
        if self.df_params['Reliable'].item()!='Y':
            ax.text(0.5,0.5, self.df_params['Reliable'].item(), transform=ax.transAxes,
                fontsize=10, va='center', ha='center')
        
        if np.isfinite(SNR2)==False:
            pass
        else:
            A21,A22,V21,V22,S21,S22,B2 = self.df.loc[0,['A21','A22','V21','V22','S21','S22','B2']]
            model_narw = gauss(xs, A21,V21,S21)+B2/2
            model_brod = gauss(xs, A22,V22,S22)+B2/2
            model_totl = model_narw+model_brod
            ax.plot(xs, model_narw, color='tab:blue'  ,       label=r'$\sigma$={:.1f}'.format(S21))
            ax.plot(xs, model_brod, color='tab:orange',       label=r'$\sigma$={:.1f}'.format(S22))
            ax.plot(xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
            ax.legend(title='S/N={:.0f}'.format(SNR2), loc='upper right')
            # text = 'sn/sb=\n'+r'{:.2f}$\pm{:.2f}$'.format(self.df_params.loc[0,'sn/sb'], self.df_params.loc[0,'e_sn/sb'])
            # ax.text(0.99,0.5,text, 
            #         transform=ax.transAxes, va='center', ha='right')
            
            # ax.text(0.99,0.5,'a21/amp={:.3e}'.format(A21/(A21+A22)), transform=ax.transAxes, ha='right')
            
            residual = self.y - (gauss(self.x,A21,V21,S21)+gauss(self.x,A22,V22,S22)+B2)
            
            ax_2Gres.axhline(0, color='gray', alpha=0.5)
            ax_2Gres.yaxis.set_tick_params(  labelleft=False)
            ax_2Gres.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
            ax_2Gres.plot(self.x, residual, color='tab:blue')
            
            chisq = np.sum((residual/self.e_y)**2)
            dof   = len(self.y)-len(self.params_2gfit)
            chisq_red = chisq/dof
            ax_2Gres.text(0.99,0.99, r'$\chi^2_\mathrm{red}$='+f'{chisq_red:.1f}', va='top',ha='right', transform=ax_2Gres.transAxes)

            F, crit = self.do_Ftest()            
            ax_2Gres.text(0.01,0.99, r'F='+f'{F:.2f}({crit:.2f})', va='top',ha='left', transform=ax_2Gres.transAxes)
            
            noise = np.std(residual)
            ax_2GFIT.axhspan(-noise, noise, color='gray', alpha=0.3    , zorder=0)
            ax_2GFIT.axhspan(-3*noise, 3*noise, color='gray', alpha=0.3, zorder=0)
            
            ax_2Gres.text(0.99,0.01, r'RMS='+f'{noise*1000:.2f} mJy', va='bottom',ha='right', transform=ax_2Gres.transAxes)
            
    def makeplot_GFIT_resampled(self, ax):
        xs = np.linspace(self.xmin, self.xmax, 1000)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(  labelleft=False)
        
        #plot G2
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        
        argwhere_A21 = np.argwhere(self.params_2gfit=='A21').item()
        argwhere_A22 = np.argwhere(self.params_2gfit=='A22').item()
        argwhere_S21 = np.argwhere(self.params_2gfit=='S21').item()
        argwhere_S22 = np.argwhere(self.params_2gfit=='S22').item()
        
        alpha = np.max([1/self.nsample_resample,1/510.])
        
        for i in range(self.nsample_resample):
            
            A21 = self.resampled[i,argwhere_A21]
            A22 = self.resampled[i,argwhere_A22]
            S21 = self.resampled[i,argwhere_S21]
            S22 = self.resampled[i,argwhere_S22]
            if 'V21' not in self.params_2gfit: V21 = self.df['V21'].item()
            else: V21 = self.resampled[i,np.argwhere(self.params_2gfit=='V21').item()]
            if 'V22' not in self.params_2gfit: V22 = self.df['V22'].item()
            else: V22 = self.resampled[i,np.argwhere(self.params_2gfit=='V22').item()]
            if 'B2' not in self.params_2gfit: B2 = self.df['B2'].item()
            else: B2 = self.resampled[i,np.argwhere(self.params_2gfit=='B2').item()]
        
            model_narw = gauss(xs, A21,V21,S21)+B2/2
            model_brod = gauss(xs, A22,V22,S22)+B2/2
            model_totl = model_narw+model_brod
            ax.plot(xs, model_narw, alpha=alpha, lw=0.5, zorder=0, color='tab:blue'  , label=r'$\sigma$={:.1f}'.format(S21))
            ax.plot(xs, model_brod, alpha=alpha, lw=0.5, zorder=2, color='tab:orange', label=r'$\sigma$={:.1f}'.format(S22))
            ax.plot(xs, model_totl, alpha=alpha, lw=0.5, zorder=1, color='gray', label=r'$\Sigma$')
        
    def makeplot_disphist(self, ax:plt.Axes) -> None:
        
        ax.set_box_aspect(1)
        # sns.kdeplot(self.list_disps, color='tab:gray', ax=ax)
        
        ax.plot(self.kde_x,self.kde_y,color='tab:gray')
        ax.set_ylim(bottom=0)
        
        # argwhere = np.argwhere(kde(xs)>0.0001)[0]
        # ax.axvline(xs[argwhere])
        
        ax2 = ax.twinx()
        bins = np.arange(0,self.list_disp.max(),1)
        sns.histplot(self.list_disp, bins=bins, color='tab:gray', alpha=0.5, label='10.3 kms', ax=ax2, edgecolor=None)
                
        # ax.hist(self.list_disp, bins=bins, color='tab:gray', alpha=0.5, label='10.3 kms')
        ax.set_xlim(0,np.max([np.percentile(self.list_disp, 99), 30]))
        
        S1 = self.df['S1'].item()
        ax.axvline(S1, color='black')
        ymax = ax.get_ylim()[1]
        ax.text(S1+0.5,ymax*0.98, r'$\sigma_\mathrm{1G}$'+f'\n{S1:.1f}', ha='left', va='top')
        
        
        percentiles = [50,60,70,80,90,95,96,97,98,99]
        vals_percentile = np.percentile(self.list_disp,percentiles)
        for i,val in enumerate(vals_percentile):
            ax.plot([val,val],[ymax*0.5,ymax*0.55], color='black')
            ax.text(val, ymax*0.551, f'{percentiles[i]:.0f}', va='bottom', ha='center')
        # ax.axvline(np.percentile(self.list_disps,95),   color='blue')
        # ax.axvline(np.percentile(self.list_disps,99),   color='orange')
        # ax.axvline(np.percentile(self.list_disps,99.9), color='red')
        
        if np.isfinite(self.df.loc[0,'SNR2']):
            ax.axvspan(self.dispmin,self.dispmax, color='lightgray', alpha=0.5)
            S21, S22 = self.df.loc[0,['S21','S22']]
            ax.axvline(S21, color='tab:blue')
            ax.axvline(S22, color='tab:orange')
            ax.text(S21,ymax*1.01, r'$\sigma_\mathrm{n}$'+'\n{:.2f}'.format(S21), ha='right', va='bottom')
            ax.text(S22,ymax*1.01, r'$\sigma_\mathrm{b}$'+'\n{:.2f}'.format(S22), ha='left',  va='bottom')
            ax.text(ax.get_xlim()[1],ymax*1.01, r'$\sigma_\mathrm{b}-\sigma_\mathrm{n}$'+'\n{:.2f}'.format(S22-S21), ha='right', va='bottom')
            
            text  = 'Bounds'
            # text += '\ndelta_disp={:.2f}'.format(self.gmodel_2GFIT.delta_disp)
            text += '\nS21=({:.2f},{:.2f})'.format(self.dispmin,self.dispmax)
            text += '\nS22=({:.2f},{:.2f})'.format(self.dispmin,self.dispmax)
            
            ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
        else:
            text  = 'Bounds'
            text += '\nS1=({:.2f},{:.2f})'.format(self.dispmin,self.dispmax)
            ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
            
        ax.set_xlabel(r'$\mathrm{km \, s^{-1}}$')
        
    def makeplot_atlas(self) -> None:
        
        from matplotlib.patches import Rectangle
        
        def dict_coord_to_subplot_coord(dict_coord):
            return [dict_coord['l'],dict_coord['b'],dict_coord['r']-dict_coord['l'],dict_coord['t']-dict_coord['b']]
        
        def paste_image(path_image, dict_coord):
            
            while os.path.exists(path_image)==False:
                os.sleep(1)
                
            frontImage = Image.open(path_image)
            
            width_pix = int(dict_coord['r']-dict_coord['l'])
            heigt_pix = int(dict_coord['t']-dict_coord['b'])
            
            frontImage_red = frontImage.resize((width_pix,heigt_pix))
            background.paste(frontImage_red, [int(dict_coord['l']),int(bgheight-dict_coord['t'])], frontImage_red) 
            os.remove(path_image)

            return
        
        suffix = self.suffix if self.suffix!='' else '_'
        savename = self.path_plot / "Plotfit_atlas{}_{}.png".format(suffix,self.name_cube)

        fig = plt.figure(figsize=(20,12))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        fig.savefig(savename)
        background = Image.open(savename)
        bgwidth, bgheight = background.size
        
        # ax_guide = fig.add_subplot([0,0,1,1])
        # ax_guide.axis('off')
        # for tick in np.arange(0,1,0.1):
        #     ax_guide.plot([tick,tick],[0,1], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.text(tick,0.95,f'{tick:.1f}',transform=ax_guide.transAxes)
        #     ax_guide.plot([0,1],[tick,tick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.text(0.95,tick,f'{tick:.1f}',transform=ax_guide.transAxes)
        
        #==================================
        coord_title = {
            'l':0.06,
            'r':0.23,
            't':0.25,
            'b':0.10,
        }
        
        coord = coord_title
        ax_title = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']])
        ax_title.axis('off')
        rect = Rectangle(xy=(0,0), width=1, height=1, transform=ax_title.transAxes, facecolor='none', edgecolor='black', linewidth=5)
        ax_title.add_patch(rect)
        
        ax_title.text(0.5, 0.65, self.name_cube, va='center', ha='center', fontsize=30, transform=ax_title.transAxes)
        ax_title.text(0.5, 0.30, self.suffix,    va='center', ha='center', fontsize=15, transform=ax_title.transAxes)
        #==================================
       
        
        #===========
        coord_frame_1GFIT = {'t':0.93,
                    'l':0.62,       'r':0.775,
                             'b':0.61 }
        coord_frame_1Gres = {'t':coord_frame_1GFIT['b']-0.01,
                    'l':coord_frame_1GFIT['l'],       'r':coord_frame_1GFIT['r'],
                             'b':0.55 }
        
        ax_1GFIT = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_1GFIT))
        ax_1Gres = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_1Gres), sharex=ax_1GFIT)
        
        self.makeplot_1GFIT(ax_1GFIT, ax_1Gres)
        
        coord_frame_2GFIT = {'t':coord_frame_1GFIT['t'],
                    'l':coord_frame_1GFIT['r']+0.01,       'r':coord_frame_1GFIT['r']*2-coord_frame_1GFIT['l']+0.01,
                             'b':coord_frame_1GFIT['b'] }
        coord_frame_2Gres = {'t':coord_frame_2GFIT['b']-0.01,
                    'l':coord_frame_2GFIT['l'],       'r':coord_frame_2GFIT['r'],
                             'b':coord_frame_1Gres['b'] }
        ax_2GFIT = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_2GFIT), sharex=ax_1GFIT, sharey=ax_1GFIT)
        ax_2Gres = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_2Gres), sharex=ax_1GFIT, sharey=ax_1Gres)
        
        self.makeplot_2GFIT(ax_2GFIT, ax_2Gres)
        
        
        
        #===============
        coord_disphist = {
            'l':0.07,
            'r':0.23,
            't':0.95,
            'b':0.65,
        }
        
        coord = coord_disphist
        ax_disphist = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']])

        ax = ax_disphist
        self.makeplot_disphist(ax)

        #===============
        
        background = Image.open(savename)
        bgwidth, bgheight = background.size
        
        coord_corner_emcee = {
                'l':0.26*bgwidth,
                'r':0.57*bgwidth,
                't':0.98*bgheight,
                'b':0.50*bgheight,
            }
        
        coord_corner_resample = {
                'l':coord_corner_emcee['l'],
                'r':coord_corner_emcee['r'],
                't':0.53*bgheight,
                'b':0.05*bgheight,
            }
        
        coord_walks = {
                'l':0.57*bgwidth,
                'r':0.98*bgwidth,
                't':coord_corner_resample['t']+0.01,
                'b':coord_corner_resample['b'],
            }
        
        coord_GFIT_resample = {
                'l':0.46,
                'r':0.555,
                't':0.48,
                'b':0.32,
            }
        
        if np.isfinite(self.df_params['e_sn'].item()):
            ax_GFIT_resamp = fig.add_subplot(dict_coord_to_subplot_coord(coord_GFIT_resample))
            self.makeplot_GFIT_resampled(ax_GFIT_resamp)
        
        fig.savefig(savename)
            
        background = Image.open(savename)
        bgwidth, bgheight = background.size
        
        if(np.isfinite(self.df['SNR2'].item())):
            self.makeplot_corner_mcmc(savefig=True)
            try: paste_image(self.savename_corner_emcee, coord_corner_emcee)
            except AttributeError: pass
            
        if self.df_params['Reliable'].item()=='Y':
            self.makeplot_walks(savefig=True)
            try: paste_image(self.savename_walks, coord_walks)
            except AttributeError: pass
            
        if 'sn' in self.df_params:
            self.makeplot_corner_resample(savefig=True)
            try: paste_image(self.savename_corner_resample, coord_corner_resample)
            except AttributeError or FileNotFoundError: pass
            
        background.save(savename, format="png")
        
        return
        
    
    def tidyup(self) -> None:
        pass
    
    
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
        
        if np.all(self.y==0) or np.all(np.isfinite(self.y)==False):
            self.df_params['Reliable']='0stacked'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Nothing seem to be stacked.")
            return False
        if len(self.list_disp)<2:
            self.df_params['Reliable']='0stacked'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; (Almost) Nothing seem to be stacked.")
            return False

        return True
    
    def check_list_disp(self):
        
        self.min_list_disp = np.min(self.list_disp)
        # self.max_list_disps = np.percentile(self.list_disps, 99)#np.max(self.list_disps)
        self.max_list_disp = np.percentile(self.list_disp, 95)#np.max(self.list_disps)
        
        # self.dispmin = np.min([np.max([np.min(self.list_disps) - self.chansep/2., self.chansep/2.]), np.min(self.list_disps)])
        # self.dispmin = np.min(self.list_disps) # - self.chansep/4.
        # self.dispmax = np.max([100,self.max_list_disps]) #np.max(self.list_disps) + self.chansep/4.
        #====
        kde = gaussian_kde(self.list_disp)#, bw_method='scott')
        kde.set_bandwidth(bw_method=kde.factor * 0.5 )
        xs = np.linspace(0,100,10000)
        
        self.kde_y = kde(xs)
        self.kde_x = xs
        
        # argwhere = np.argwhere(self.kde_y>0.0001)[0].item()
        # self.dispmin = np.max([self.kde_x[argwhere],0])
        # # self.dispmin = self.min_list_disps-self.chansep/4.
        
        # argwhere = np.argwhere(self.kde_y>0.0001)[-1].item()
        # # self.dispmax = np.max([100,self.max_list_disps])#self.kde_x[argwhere]
        # # self.dispmax = np.max(self.list_disps)#+self.chansep/4
        # self.dispmax = self.kde_x[argwhere]
        
        # argmin = np.argmin(self.list_disp)
        # argmax = np.argmax(self.list_disp)
        # self.dispmin = self.list_disp[argmin] - self.e_list_disp[argmin]
        # self.dispmax = self.list_disp[argmax] + self.e_list_disp[argmax]
        
        self.dispmin = np.min(self.list_disp-self.e_list_disp)
        self.dispmax = np.max(self.list_disp+self.e_list_disp)
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
            
        # if self.dispmax-self.dispmin<self.chansep/2:
        #     self.df_params['Reliable']='disprng'
        #     self.makeplot_atlas()
        #     print(f"[Plotfit] {self.name_cube:>{self.longest_length}} pass; Range of stacked dispersion is narrow")
        #     return False
        
        # self.percentile_maxdisp = percentile_maxdisp
        
        return True
    
    # def redo(self):
    #     argwhere_S21 = np.argwhere(self.params_2gfit=='S21').item()
    #     argwhere_S22 = np.argwhere(self.params_2gfit=='S22').item()
        
    #     while True:
    #         skew_S21 = skew(self.flat_samples[:,argwhere_S21])
    #         skew_S22 = skew(self.flat_samples[:,argwhere_S22])
            
    #         if skew_S21<1 and skew_S22>-1:
    #             break
            
    #         if skew_S21>1 and skew_S22>-1:
    #             self.df_params['Reliable'] = 'disprng_S21'
    #             self.GFIT2_success = False
    #             break
            
    #         if skew_S22<-1:
    #             if self.percentile_maxdisp<100:
    #                 pass
    #             else:
    #                 if skew_S21<1: self.df_params['Reliable'] = 'disprng_S22'
    #                 if skew_S21>1: self.df_params['Reliable'] = 'disprng_both'
    #                 self.GFIT2_success = False
    #                 break
            
    #         if skew_S22<-1:
    #             self.percentile_maxdisp+=10
    #             if self.percentile_maxdisp<100:
    #                 self.max_list_disps = np.percentile(self.list_disps, self.percentile_maxdisp)
    #             else:
    #                 self.max_list_disps = self.dispmax

    #         # if S21_skewed_left: 
    #         #     self.min_list_disps -= np.max([self.chansep/2.,0])

    #         print(f'[Plotfit] {self.name_cube:>{self.longest_length}} redo')
    #         self.df_params['Reliable']='N'
    #         self.makefit_2G()
    #         self.makeplot_atlas()
            
    #         # if (self.min_list_disps<=0 or self.min_list_disps<) and self.max_list_disps>self.dispmax+self.chansep/2.:
    #         #     break
            
    #     return
    
    def evaluate_2GFIT(self):
        
        def sort_outliers(array, weight=1.5):
            
            perc25, perc75 = np.nanpercentile(array, [25,75])
            IQR = perc75-perc25
            
            end_lower = perc25 - IQR*weight
            end_highr = perc75 + IQR*weight
            
            return array[(array>end_lower) & (array<end_highr)]
            
        
        A21,A22,N2 = self.df.loc[0,['A21','A22','N2']]
        
        if A21<3*N2:
            self.df_params['Reliable'] = 'lowA21'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling is a no go; A21 meaningless.")
            self.GFIT2_success = False
            return
        if A22<3*N2:
            self.df_params['Reliable'] = 'lowA22'
            print(f"[Plotfit] {self.name_cube:>{self.longest_length}}'s resampling is a no go; A22 meaningless.")
            self.GFIT2_success = False
            return
        
        for param in ['A21','A22','S21','S22']:
            argwhere_param = np.argwhere(self.params_2gfit==param).item()
            params = self.flat_samples[:,argwhere_param]
            
            # perc_param = np.percentile(params, (2.5,97.5))
            # params = params[(params>perc_param[0])&(params<perc_param[1])]
            
            params = sort_outliers(params)
            
            skew_param = skew(params)
            self.df[f'Skew_{param}'] = skew_param

        skew_A21 = self.df['Skew_A21'].item()
        skew_A22 = self.df['Skew_A22'].item()
        skew_S21 = self.df['Skew_S21'].item()
        skew_S22 = self.df['Skew_S22'].item()
        
        if skew_S21>0.6 and not skew_S22<-0.6:
            self.df_params['Reliable'] = 'S21_at_edge'
            self.GFIT2_success = False
            return
            
        if not skew_S21>0.6 and skew_S22<-0.6:
            self.df_params['Reliable'] = 'S22_at_edge'
            self.GFIT2_success = False
            return
            
        if skew_S21<-0.6 and skew_S22>0.6:
            self.df_params['Reliable'] = 'S21~S22'
            self.GFIT2_success = False
            return
        
        if np.abs(skew_S21)>.6 or np.abs(skew_S22)>.6:
            self.df_params['Reliable'] = 'disps_at_edge'
            self.GFIT2_success = False
            return
            
        if np.abs(skew_A21)>.6 or np.abs(skew_A22)>.6:
            self.df_params['Reliable'] = 'ampls_at_edge'
            self.GFIT2_success = False
            
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
        
        self.makefit_1G(self.x, self.y, self.e_y, mcmc=True)
        self.limit_range(multiplier_disp=15)
        self.makefit_1G(self.x, self.y, self.e_y, mcmc=True)
        # self.makefit_1G(self.x, self.y, self.e_y, mcmc=True)
        self.makeplot_atlas()
        
        self.makefit_2G()
        self.makeplot_atlas()
        
        # if self.GFIT2_success:
        #     self.redo()
        #     self.makeplot_atlas()
        
        # if self.GFIT2_success:
        #     self.evaluate_2GFIT()
        #     self.makeplot_atlas()
        
        # if self.GFIT2_success:           
        #     self.resample(nsample=nsample_resample, pbar_resample=pbar_resample)
        #     # self.test_skew()
        #     self.makeplot_atlas()
        
        # self.writestat('Finished.0')
        self.removestat()
        

