
import os

import numpy as np
import seaborn as sns
from chainconsumer import Chain, ChainConsumer, Truth
from matplotlib.patches import Rectangle
from PIL import Image, ImageFile

from.subroutines_Plotfitter import gauss, sort_outliers, gaussian_area, _idx, _softplus, _inv_softplus, _sigmoid_mapped, demap_params_unconstrained, relabel_by_width
import pandas as pd
import pylab as plt
import time
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Plotter:
    
    def __init__(self, path_plot, name_cube, suffix, list_disp, df, df_params, gmodel=None, sampler=None, resampled=None, unconstrained=False):
        
        self.path_plot = path_plot
        self.name_cube = name_cube
        self.suffix    = suffix
        
        self.list_disp = list_disp
        self.df        = df
        self.df_params = df_params
        
        self.gmodel    = gmodel
        self.sampler   = sampler
        self.resampled = resampled
        
        self.savename_autocorr        = path_plot / 'Plotfit_autocorr_{}.png'.format(self.name_cube)
        self.savename_corner_emcee    = path_plot / "emcee_corner{}_{}.png".format(self.suffix, self.name_cube)
        self.savename_corner_resample = path_plot / "resampled_corner{}_{}.png".format(self.suffix, self.name_cube)
        self.savename_walks           = path_plot / "Plotfit_walks{}_{}.png".format(self.suffix, self.name_cube)
        self.savename_paramshist      = path_plot / "Plotfit_params{}_{}.png".format(self.suffix, self.name_cube)
        self.savename_atlas           = path_plot / "Plotfit_atlas{}_{}.png".format(suffix,self.name_cube)
        
        # self.savename_GFIT            = path_plot / "Plotfit_{{}}GFIT{}_{}.png".format(suffix,self.name_cube)
        
        self.xs = np.linspace(self.gmodel.x.min(), self.gmodel.x.max(), 1000)
        
        self.x   = self.gmodel.x
        self.y   = self.gmodel.y
        self.e_y = self.gmodel.e_y
        
        if sampler is not None:
            tau = sampler.get_autocorr_time(tol=0)
            self.burnin = int(2   * np.nanmax(tau))
            self.thin   = int(0.5 * np.nanmin(tau))
            
        self.names_param = np.array(gmodel.names_param)
        self.unconstrained = unconstrained
        
        return
    
    def makeplot_GFIT(self, G, savefig:bool=True):
        
        fig, axs = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [5,1]})
        
        ax = axs[0]
        
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        
        for g in range(1,G+1):
            if G==1: AA,VV,SS,BB = self.df.loc[0,[f'A{g}',f'V{g}',f'S{g}',f'B{g}']]
            else:
                Gg = f'{G}{g}'
                AA,VV,SS,BB = self.df.loc[0,[f'A{Gg}',f'V{Gg}',f'S{Gg}',f'B{G}']]
            model = gauss(self.xs,AA,VV,SS)+BB
            ax.plot(self.xs, model, label=r'$\sigma$={:.1f}'.format(SS))
        
        model_totl = np.sum([gauss(self.xs,self.df.loc[0,f'A{G}{g}'],self.df.loc[0,f'V{G}{g}'],self.df.loc[0,f'S{G}{g}']) for g in range(1,G+1)],axis=0)+self.df.loc[0,f'B{G}']
        ax.plot(self.xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
        ax.legend()
        
        if G==1:
            residual = self.y - (gauss(self.x,self.df.loc[0,f'A{G}'],self.df.loc[0,f'V{G}'],self.df.loc[0,f'S{G}']) + self.df.loc[0,f'B{G}'])
        else:
            residual = self.y - (np.sum([gauss(self.x,self.df.loc[0,f'A{G}{g}'],self.df.loc[0,f'V{G}{g}'],self.df.loc[0,f'S{G}{g}']) for g in range(1,G+1)],axis=0)+self.df.loc[0,f'B{G}'])
        
        #plot G2
        ax = axs[1]
        ax.axhline(0, color='gray', alpha=0.5)
        ax.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
        ax.scatter(self.x, residual, s=3, color='tab:blue')
        
        fig.savefig(self.path_plot / "Plotfit_{}GFIT{}_{}.png".format(G,self.suffix,self.name_cube))
        

    def makeplot_corner_emcee(self, savefig:bool=True) -> plt.Figure:
        
        flat_samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)
        if self.unconstrained:
            flat_samples = demap_params_unconstrained(flat_samples, self.gmodel)
        df = pd.DataFrame()
        for i, label in enumerate(self.names_param):
            df[label] = flat_samples[:,i]
        chain = Chain(samples=df, name='emcee')
        consumer = ChainConsumer().add_chain(chain)        
        
        try:
            fig = consumer.plotter.plot()
            
        except:
            return
        
        # iA21 = _idx(self.names_param, 'A21')
        # iA22 = _idx(self.names_param, 'A22')
        # iS21 = _idx(self.names_param, 'S21')
        # iS22 = _idx(self.names_param, 'S22')
        
        # if 'Skew_A21' in self.df:
        #     N = len(self.names_param)
        #     ax = fig.axes[iA21*(N+1)]; ax.text(0.98,0.98, 'Skew={:.1e}\nKurt={:.1e}'.format(self.df['Skew_A21'].item(),self.df['Kurt_A21'].item()), va='top', ha='right', fontsize=10, transform=ax.transAxes)
        #     ax = fig.axes[iA22*(N+1)]; ax.text(0.98,0.98, 'Skew={:.1e}\nKurt={:.1e}'.format(self.df['Skew_A22'].item(),self.df['Kurt_A22'].item()), va='top', ha='right', fontsize=10, transform=ax.transAxes)
        #     ax = fig.axes[iS21*(N+1)]; ax.text(0.98,0.98, 'Skew={:.1e}\nKurt={:.1e}'.format(self.df['Skew_S21'].item(),self.df['Kurt_S21'].item()), va='top', ha='right', fontsize=10, transform=ax.transAxes)
        #     ax = fig.axes[iS22*(N+1)]; ax.text(0.98,0.98, 'Skew={:.1e}\nKurt={:.1e}'.format(self.df['Skew_S22'].item(),self.df['Kurt_S22'].item()), va='top', ha='right', fontsize=10, transform=ax.transAxes)
        
        if savefig:
            fig.savefig(self.savename_corner_emcee, transparent=True)
        
        return fig
        
    def makeplot_corner_resample(self, savefig:bool=True) -> plt.Figure:
        
        if ~np.isfinite(self.df_params['e_sn'].item()): return
        df = pd.DataFrame()
        
        for i, label in enumerate(self.names_param):
            df[label] = self.resampled[:,i]
        
        chain = Chain(samples=df, name='Resampled')
        consumer = ChainConsumer().add_chain(chain)
        
        dict_truth = {}
        for param in self.names_param:
            dict_truth[param] = self.df[param].item()
        consumer.add_truth(Truth(location=dict_truth, name='emcee', color='tab:blue'))
        dict_truth = {}
        for i, param in enumerate(self.names_param):
            dict_truth[param] = np.nanmedian(sort_outliers(self.resampled[:,i]))
        consumer.add_truth(Truth(location=dict_truth, name='resampled', color='tab:orange',line_style=':'))
        
        try:
            fig = consumer.plotter.plot()
            axs = [fig.get_axes()[i*len(self.names_param)] for i in range(len(self.names_param))]
            for ax in axs:
                ax.set_ylabel('') 
        except: #Exception as e:
            # print(e)
            return
        
        if savefig:
            fig.savefig(self.savename_corner_resample, transparent=True)
        
        return fig
    
    def makeplot_walks(self, savefig:bool=True, transparent:bool=True) -> plt.Figure:
        
        if ~np.isfinite(self.df_params['e_sn'].item()): return
        if ~np.isfinite(self.df['SNR2'].item()): return
        
        df = pd.DataFrame()
        for i, label in enumerate(self.names_param):
            df[label] = self.resampled[:,i]
                        
        chain = Chain(samples=df, name='Resampled')
        consumer = ChainConsumer().add_chain(chain)

        try:
            fig = consumer.plotter.plot_walks(convolve=100, plot_weights=False, figsize=(6,8))
        except:
            return
        
        if savefig:
            fig.savefig(self.savename_walks, transparent=transparent)
        
        return fig
    
    
    def makeplot_GFIT_atlas(self, G, ax_GFIT, ax_Gres, config=None):
                
        if config=='right':
            ax_GFIT.xaxis.set_tick_params(labelbottom=False)
            ax_GFIT.yaxis.set_tick_params(  labelleft=False)
            ax_Gres.yaxis.set_tick_params(  labelleft=False)
        
        ax = ax_GFIT
        ax.set_title(f'{G}G', fontsize=20)
        ax.xaxis.set_tick_params(labelbottom=False)
        
        ax.axhline(0,color='gray',alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        
        SNRG = self.df.loc[0,f'SNR{G}']
        if not np.isfinite(SNRG): return

        for g in range(1,G+1):
            if G==1: AA,VV,SS,BB = self.df.loc[0,[f'A{g}',f'V{g}',f'S{g}',f'B{g}']]
            else:
                Gg = f'{G}{g}'
                AA,VV,SS,BB = self.df.loc[0,[f'A{Gg}',f'V{Gg}',f'S{Gg}',f'B{G}']]
            model = gauss(self.xs,AA,VV,SS)+BB
            ax.plot(self.xs, model, label=r'$\sigma$={:.1f}'.format(SS))
        
        if G==1:
            model_totl = gauss(self.xs,self.df.loc[0,f'A{G}'],self.df.loc[0,f'V{G}'],self.df.loc[0,f'S{G}']) + self.df.loc[0,f'B{G}']
        else:
            model_totl = np.sum([gauss(self.xs,self.df.loc[0,f'A{G}{g}'],self.df.loc[0,f'V{G}{g}'],self.df.loc[0,f'S{G}{g}']) for g in range(1,G+1)],axis=0)+self.df.loc[0,f'B{G}']
        ax.plot(self.xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
        ax.legend(title='S/N={:.0f}'.format(SNRG), loc='upper right')

        if G==1:
            model_totl = gauss(self.x,self.df.loc[0,f'A{G}'],self.df.loc[0,f'V{G}'],self.df.loc[0,f'S{G}']) + self.df.loc[0,f'B{G}']
        else:
            model_totl = np.sum([gauss(self.x,self.df.loc[0,f'A{G}{g}'],self.df.loc[0,f'V{G}{g}'],self.df.loc[0,f'S{G}{g}']) for g in range(1,G+1)],axis=0)+self.df.loc[0,f'B{G}']
        residuals = self.y - model_totl
    
        #plot G2
        ax = ax_Gres
        ax.axhline(0, color='gray', alpha=0.5)
        ax.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
        ax.scatter(self.x, residuals, s=3, color='tab:blue')
            
        chisq = np.sum((residuals/self.e_y)**2)
        dof   = len(self.y)-len(self.names_param)
        chisq_red = chisq/dof
        ax_Gres.text(0.99,0.99, r'$\chi^2_\mathrm{red}$='+f'{chisq_red:.1f}', va='top',ha='right', transform=ax_Gres.transAxes)

        # F, crit = self.df.loc[0,['F-test','F-crit']]        
        # ax_2Gres.text(0.01,0.99, r'F='+f'{F:.2f}({crit:.2f})', va='top',ha='left', transform=ax_2Gres.transAxes)
        
        NG,BG = self.df.loc[0,[f'N{G}',f'B{G}']]
        
        # Noise
        ax_GFIT.axhspan(  -NG+BG,   NG+BG, color='gray', alpha=0.2, zorder=0)
        ax_GFIT.axhspan(-3*NG+BG, 3*NG+BG, color='gray', alpha=0.2, zorder=0)
        
        ax_Gres.text(0.99,0.01, r'RMS='+f'{NG*1000:.2f} mJy', va='bottom',ha='right', transform=ax_Gres.transAxes)

            
    def makeplot_GFIT_resampled(self, ax):
        
        if 'A21' in self.names_param: G=2
        if 'A31' in self.names_param: G=3
        
        self.nsample_resample = len(self.resampled[:,0])
        alpha = np.max([1/self.nsample_resample,1/510.])
        
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(  labelleft=False)
        
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        
        colors = ['tab:blue','tab:orange','tab:green']
        
        for i in range(self.nsample_resample):
            
            if np.isin(f'B{G}',self.names_param):
                iB = _idx(self.names_param,f'B{G}')
                BB = self.resampled[i,iB]
            else:
                BB = self.df.loc[0,f'B{G}']
            
            for g in range(1,G+1):
                Gg = f'{G}{g}'
                iA = _idx(self.names_param,f'A{Gg}')
                iS = _idx(self.names_param,f'S{Gg}')
                
                AA = self.resampled[i,iA]
                SS = self.resampled[i,iS]
                
                if np.isin(f'V{Gg}',self.names_param):
                    iV = _idx(self.names_param,f'V{Gg}')
                    VV = self.resampled[i,iV]
                else:
                    VV = self.df.loc[0,f'V{Gg}']
                    
                model = gauss(self.xs, AA,VV,SS)+BB
                    
                ax.plot(self.xs, model, alpha=alpha, lw=0.5, zorder=0, color=colors[g-1])
        
        
    def makeplot_disphist(self, ax:plt.Axes) -> None:
        
        if len(self.list_disp)<2: return
        
        ax.set_box_aspect(1)
        
        # ax.plot(self.kde_x,self.kde_y,color='tab:gray')
        ax.set_ylim(bottom=0)
        
        ax2 = ax.twinx()
        bins = np.arange(0,self.list_disp.max(),1)
        sns.histplot(self.list_disp, bins=bins, color='tab:gray', alpha=0.5, label='10.3 kms', ax=ax2, edgecolor=None, kde=True)
                
        S1 = self.df['S1'].item()
        ax.axvline(S1, color='black')
        ymax = ax.get_ylim()[1]
        ax.text(S1+0.5,ymax*0.98, r'$\sigma_\mathrm{1G}$'+f'\n{S1:.1f}', ha='left', va='top')
        
        ax.set_xlim(0,np.max([np.percentile(self.list_disp, 99), S1*3]))
        
        percentiles = [50,60,70,80,90,95,96,97,98,99]
        vals_percentile = np.percentile(self.list_disp,percentiles)
        for i,val in enumerate(vals_percentile):
            ax.plot([val,val],[ymax*0.5,ymax*0.55], color='black')
            ax.text(val, ymax*0.551, f'{percentiles[i]:.0f}', va='bottom', ha='center')
            
        ax.set_xlabel(r'$\mathrm{km \, s^{-1}}$')
        
        G = None
        if np.isin('A21',self.names_param): G=2
        if np.isin('A31',self.names_param): G=3
        if G is None: return
        
        if not np.isfinite(self.df.loc[0,f'SNR{G}']): return

            # ax.axvspan(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S22'][1], color='lightgray', alpha=0.5)
        
        for g in range(1,G+1):
            param = f'S{G}{g}'
            SS = self.df.loc[0,param]
            ax.axvline(SS)
            ax.text(SS,ymax*1.01, r'{param}'+'\n{:.2f}'.format(SS), ha='right', va='bottom')
            
        # ax.text(ax.get_xlim()[1],ymax*1.01, r'$\sigma_\mathrm{b}-\sigma_\mathrm{n}$'+'\n{:.2f}'.format(S22-S21), ha='right', va='bottom')
            
            # text  = 'Bounds'
            # text += '\nS21=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S21'][1])
            # text += '\nS22=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S22'][0],self.gmodel.dict_bound['S22'][1])
            
            # ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
        # else:
        #     text  = 'Bounds'
        #     text += '\nS1=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S21'][1])
        #     ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
            
        
        
    def makeplot_paramshist(self, key, savefig:bool=True, transparent:bool=True) -> None:
        
        if 'A31' in self.names_param: G = 3
        if 'A21' in self.names_param: G = 2
            
        sn = self.resampled[:,np.argwhere(self.names_param==f'S{G}1').item()]
        sb = self.resampled[:,np.argwhere(self.names_param==f'S{G}2').item()]
        An = gaussian_area(self.resampled[:,np.argwhere(self.names_param==f'A{G}1').item()], sn)
        Ab = gaussian_area(self.resampled[:,np.argwhere(self.names_param==f'A{G}2').item()], sb)
        At = An+Ab
        
        dict_resampled = {
            'sn':sn,
            'sb':sb,
            'An':An,
            'Ab':Ab,
            'At':At,
            
            'sn/sb':sn/sb,
            'An/At':An/At,
            'log(sb-sn)':np.log10(sb-sn),
        }
        
        for keyy in dict_resampled.keys():
            dict_resampled[keyy] = sort_outliers(dict_resampled[keyy])
        
        # dict_resampled['sn/sb'] = dict_resampled['sn/sb'][dict_resampled['sn/sb']<0.9]
        # print(dict_resampled)
        
        dict_title = {
            'sn':r'$\sigma_\mathrm{n}$',
            'sb':r'$\sigma_\mathrm{b}$',
            'An':r'$A_\mathrm{n}$',
            'Ab':r'$A_\mathrm{b}$',
            'At':r'$A_\mathrm{tot}$',
            
            'sn/sb':r'$\sigma_\mathrm{n}/\sigma_\mathrm{b}$',
            'An/At':r'$A_\mathrm{n}/A_\mathrm{tot}$',
            'log(sb-sn)':r'$\log (\sigma_\mathrm{b}-\sigma_\mathrm{n})$',
        }
        
        dict_xlim = {
            'sn/sb':[0,1],
            'An/At':[0,1],
            'log(sb-sn)':[0,2]
        }
                
        df = pd.DataFrame()
        
        df[key] = dict_resampled[key]

        chain = Chain(samples=df, name='Resampled')
        consumer = ChainConsumer().add_chain(chain)
        
        # consumer.add_truth(Truth(location={key:self.df_params[key].item()}, name='emcee', color='tab:blue'))
        # consumer.add_truth(Truth(location={key:np.median(dict_resampled[key])}, name='resampled', color='tab:orange'))
        
        try:
            fig = consumer.plotter.plot_distributions(figsize=(4,5))#columns=[dict_title[key]])
        except Exception as e:
            print(e)
            return
        
        if savefig:
            ax = fig.get_axes()[0]
            ax.set_box_aspect(1)
            
            ax.axvline(self.df_params[key].item(), color='tab:blue')
            ax.axvline(np.mean(dict_resampled[key]), color='tab:orange')
            
            ax.axvspan(self.df_params[key].item()-self.df_params['e_'+key].item(),
                       self.df_params[key].item()+self.df_params['e_'+key].item(),
                       color='tab:blue', alpha=0.1, label='emcee', zorder=0)
            ax.axvspan(np.mean(dict_resampled[key])-np.std(dict_resampled[key]),
                       np.mean(dict_resampled[key])+np.std(dict_resampled[key]),
                       color='tab:orange', alpha=0.1, label='resampled', zorder=0)
            
            ax.set_xlim(dict_xlim[key][0],dict_xlim[key][1])
            
            fig.savefig(self.savename_paramshist, transparent=True)
        
        return fig
        
    def makeplot_atlas(self, G=2) -> None:
        
        def dict_coord_to_subplot_coord(dict_coord):
            return [dict_coord['l'],dict_coord['b'],dict_coord['r']-dict_coord['l'],dict_coord['t']-dict_coord['b']]
        
        def paste_image(path_image, dict_coord):
            # Option A: skip if not there
            if not os.path.exists(path_image):
                return

            # Option B (safer): short timeout
            # t0 = time.time()
            # while not os.path.exists(path_image):
            #     if time.time() - t0 > 10:  # 10s
            #         return
            #     time.sleep(0.25)

            frontImage = Image.open(path_image)
            width_pix = int(dict_coord['r']-dict_coord['l'])
            heigt_pix = int(dict_coord['t']-dict_coord['b'])
            frontImage_red = frontImage.resize((width_pix,heigt_pix))
            background.paste(frontImage_red, [int(dict_coord['l']), int(bgheight-dict_coord['t'])], frontImage_red)
            os.remove(path_image)
        
        suffix = self.suffix if self.suffix!='' else '_'
        

        fig = plt.figure(figsize=(25,15))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        background = Image.open(buf)
        
        bgwidth, bgheight = background.size
        
        # ax_guide = fig.add_subplot([0,0,1,1])
        # ax_guide.axis('off')
        # for tick in np.arange(0,1,0.1):
        #     ax_guide.plot([tick,tick],[0,1], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.text(tick,0.95,f'{tick:.1f}',transform=ax_guide.transAxes)
        #     ax_guide.plot([0,1],[tick,tick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.text(0.95,tick,f'{tick:.1f}',transform=ax_guide.transAxes)
        # for minortick in np.arange(0,1,0.01):
        #     ax_guide.plot([minortick,minortick],[0.99,1], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([minortick,minortick],[0,0.01], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([0.99,1],[minortick,minortick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([0,0.01],[minortick,minortick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        # for minortick in np.arange(0,1,0.05):
        #     ax_guide.plot([minortick,minortick],[0.97,1], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([minortick,minortick],[0,0.03], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([0.97,1],[minortick,minortick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        #     ax_guide.plot([0,0.03],[minortick,minortick], color='gray', alpha=0.5, transform=ax_guide.transAxes)
        
        #==================================
        coord_title = {
            'l':0.07,
            'r':0.23,
            't':0.93,
            'b':0.84,
        }
        
        coord = coord_title
        ax_title = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']])
        ax_title.axis('off')
        rect = Rectangle(xy=(0,0), width=1, height=1, transform=ax_title.transAxes, facecolor='none', edgecolor='black', linewidth=5)
        ax_title.add_patch(rect)
        
        ax_title.text(0.5, 0.65, self.name_cube, va='center', ha='center', fontsize=25, transform=ax_title.transAxes)
        ax_title.text(0.5, 0.30, self.suffix,    va='center', ha='center', fontsize=15, transform=ax_title.transAxes)
        #==================================
       
        
        #===========
        coord_frame_1GFIT = {'t':0.93,
                    'l':0.60,       'r':0.765,
                             'b':0.61 }
        coord_frame_1Gres = {'t':coord_frame_1GFIT['b']-0.01,
                    'l':coord_frame_1GFIT['l'],       'r':coord_frame_1GFIT['r'],
                             'b':0.55 }
        
        ax_1GFIT = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_1GFIT))
        ax_1Gres = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_1Gres), sharex=ax_1GFIT)
        
        self.makeplot_GFIT_atlas(1, ax_1GFIT, ax_1Gres)
        
        coord_frame_2GFIT = {'t':coord_frame_1GFIT['t'],
                    'l':coord_frame_1GFIT['r']+0.01,       'r':coord_frame_1GFIT['r']*2-coord_frame_1GFIT['l']+0.01,
                             'b':coord_frame_1GFIT['b'] }
        coord_frame_2Gres = {'t':coord_frame_2GFIT['b']-0.01,
                    'l':coord_frame_2GFIT['l'],       'r':coord_frame_2GFIT['r'],
                             'b':coord_frame_1Gres['b'] }
        ax_2GFIT = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_2GFIT), sharex=ax_1GFIT, sharey=ax_1GFIT)
        ax_2Gres = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_2Gres), sharex=ax_1GFIT, sharey=ax_1Gres)
        
        self.makeplot_GFIT_atlas(G, ax_2GFIT, ax_2Gres, config='right')
        
        #===============
        coord_disphist = {
            'l':0.07,
            'r':0.23,
            't':0.83,
            'b':0.53,
        }
        
        coord = coord_disphist
        ax_disphist = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']])

        ax = ax_disphist
        self.makeplot_disphist(ax)

        #===============
        
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
                'l':0.04*bgwidth,
                'r':0.29*bgwidth,
                't':coord_corner_resample['t']+0.01*bgheight,
                'b':coord_corner_resample['b']-0.005*bgheight,
            }
        
        coord_GFIT_resample = {
                'l':0.60,
                'r':0.70,
                't':0.48,
                'b':0.32,
            }
        
        if np.isfinite(self.df_params['e_sn'].item()):
            ax_GFIT_resamp = fig.add_subplot(dict_coord_to_subplot_coord(coord_GFIT_resample))
            self.makeplot_GFIT_resampled(ax_GFIT_resamp)
        
            
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        background = Image.open(buf)
        
        # background = Image.open(self.savename_atlas)
        bgwidth, bgheight = background.size
        
        if(np.isfinite(self.df[f'SNR{G}'].item())):
            self.makeplot_corner_emcee(savefig=True)
            try: paste_image(self.savename_corner_emcee, coord_corner_emcee)
            except AttributeError: pass
            
        if 'sn' in self.df_params:
            self.makeplot_walks(savefig=True)
            try: paste_image(self.savename_walks, coord_walks)
            except AttributeError: pass
            
            
        if 'sn' in self.df_params:
            self.makeplot_corner_resample(savefig=True)
            try: paste_image(self.savename_corner_resample, coord_corner_resample)
            except AttributeError or FileNotFoundError: pass
                        
        if 'sn' in self.df_params:
            keys = ['sn/sb','An/At','log(sb-sn)']
            width = (coord_GFIT_resample['r']-coord_GFIT_resample['l'])+0.015
            height = 0.17
            offset = 0.007
            left = (coord_GFIT_resample['l'] - 0.01)*bgwidth
            for i, key in enumerate(keys):
                coord_paramshist = {
                        'l': left,
                        'r': left + (width * bgwidth),
                        't':(0.105+height)*bgheight,
                        'b':(0.06)*bgheight,
                }
                left = coord_paramshist['r']+(offset*bgwidth)
                if np.isfinite(self.df_params['e_sn'].item()):
                    self.makeplot_paramshist(key=key)
                    try: paste_image(self.savename_paramshist, coord_paramshist)
                    except AttributeError or FileNotFoundError: pass
                    
        background.save(self.savename_atlas, format="png")
        
        
        
        return