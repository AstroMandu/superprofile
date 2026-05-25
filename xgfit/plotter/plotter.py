import gc
import io
import os

import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from chainconsumer import Chain, ChainConsumer, Truth
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ..gmodel.mappers import map_params
from ..subroutines import gauss, gaussian_area, idx, sort_outliers


class Plotter:
    
    def __init__(self, path_plot, name_cube, suffix, list_disp, df, df_params, gmodel=None, sampler=None, resampled=None, burnin=0, thin=1, list_NHI_=None):
        
        self.path_plot = path_plot
        self.name_cube = name_cube
        self.suffix    = suffix
        
        self.list_disp = list_disp
        self.list_NHI_ = list_NHI_
        
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
        self.savename_atlas           = path_plot / "Plotfit_atlas_{}{}.png".format(self.name_cube,suffix)
        
        # self.savename_GFIT            = path_plot / "Plotfit_{{}}GFIT{}_{}.png".format(suffix,self.name_cube)
        
        self.xs = np.linspace(self.gmodel.x.min(), self.gmodel.x.max(), 1000)
        self.chansep = np.mean(np.abs(np.diff(self.xs)))
        
        self.x   = self.gmodel.x
        self.y   = self.gmodel.y
        self.e_y = self.gmodel.e_y
        
        self.names_param = np.array(gmodel.names_param)
        
        self.burnin = burnin
        self.thin   = thin
        
        return
    


    # def _annotate_spearman_on_chainconsumer_corner(
    #     self, fig, df, names,
    #     fmt=r"\rho_s={:+.2f}$",
    #     trim=5  # percent to trim on each side
    # ):
    #     n = len(names)
    #     axes = np.array(fig.axes).reshape((n, n))

    #     for i in range(n):
    #         for j in range(n):

    #             ax = axes[i, j]

    #             if i == j:
    #                 continue

    #             if len(ax.lines) == 0 and len(ax.collections) == 0 and len(ax.patches) == 0:
    #                 continue

    #             x = df[names[j]].to_numpy()
    #             y = df[names[i]].to_numpy()

    #             # --- finite mask ---
    #             m = np.isfinite(x) & np.isfinite(y)
    #             if m.sum() < 5:
    #                 continue

    #             x = x[m]
    #             y = y[m]

    #             # --- percentile trimming (robust tail removal) ---
    #             if trim > 0:
    #                 x_lo, x_hi = np.percentile(x, [trim, 100-trim])
    #                 y_lo, y_hi = np.percentile(y, [trim, 100-trim])

    #                 keep = (
    #                     (x >= x_lo) & (x <= x_hi) &
    #                     (y >= y_lo) & (y <= y_hi)
    #                 )

    #                 if keep.sum() < 5:
    #                     continue

    #                 x = x[keep]
    #                 y = y[keep]

    #             # --- Spearman ---
    #             rho_s, _ = spearmanr(x, y)

    #             ax.text(
    #                 0.04, 0.96,
    #                 fmt.format(rho_s),
    #                 transform=ax.transAxes,
    #                 ha="left", va="top",
    #                 fontsize=9,
    #                 bbox=dict(
    #                     boxstyle="round,pad=0.2",
    #                     facecolor="white",
    #                     alpha=0.65,
    #                     edgecolor="none"
    #                 ),
    #                 zorder=100
    #             )

    #     return fig
    
    
    def _annotate_spearman_on_chainconsumer_corner(self, fig, df, names, fmt=r"$r_s={:+.3f}$"):
        n = len(names)
        axes = np.array(fig.axes).reshape((n, n))
        
        def robust_pca_ratio(samples, trim=5):
            # trim outer percentiles
            low = np.percentile(samples, trim, axis=0)
            high = np.percentile(samples, 100-trim, axis=0)
            mask = np.all((samples >= low) & (samples <= high), axis=1)
            samples = samples[mask]

            # PCA
            C = np.cov(samples, rowvar=False)
            eigvals = np.linalg.eigvalsh(C)
            eigvals = np.sort(eigvals)[::-1]
            ratio = eigvals[0] / eigvals.sum()
            return ratio

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                if i == j:
                    continue
                if len(ax.lines) == 0 and len(ax.collections) == 0 and len(ax.patches) == 0:
                    continue

                samples_2d = df[[names[j], names[i]]].to_numpy()
                
                # ratio, total = pca_degeneracy(samples_2d)
                ratio = robust_pca_ratio(samples_2d)

                ax.text(
                    0.04, 0.96,
                    fmt.format(ratio),
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"),
                    zorder=100
                )
                
                
                
                
        return fig
    
    def _annotate_corr_on_chainconsumer_corner(
        self,
        fig,
        df,
        names,
        # fmt = r"$\mathrm{{dCor}}={:+.3f}$",
        fmt = r"$r_s={:+.3f}$",
        pos=(0.04, 0.96),  # anchor in axes coords
    ):
        n = len(names)
        axes = np.array(fig.axes).reshape((n, n))

        for i in range(n):
            for j in range(i+1,n):
                ax = axes[j,i]

                # if i == j:
                #     continue
                
                # colname_dcor = f'dcor_{names[i]}{names[j]}'
                colname_corr = f'rs_{names[i]}{names[j]}'
                colname_pcor = f'p_rs_{names[i]}{names[j]}'
                
                corr = df[colname_corr].item()
                pcor = df[colname_pcor].item()
                
                star = ''
                if pcor<0.05: 
                    star = '*'
                if pcor<0.01: 
                    star = '**'
                if pcor<0.001:
                    star = '***'

                ax.text(
                    pos[0], pos[1],
                    fmt.format(corr) + star,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.65,
                        edgecolor="none",
                    ),
                    zorder=100,
                )

        return fig

    
    
    def cleanup(self):
        try:    os.remove(self.savename_autocorr)
        except: pass
        try:    os.remove(self.savename_corner_emcee)
        except: pass
        try:    os.remove(self.savename_corner_resample)
        except: pass
        try:    os.remove(self.savename_walks)
        except: pass
        try:    os.remove(self.savename_paramshist)
        except: pass
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
                # AA,VV,SS,BB = self.df.loc[0,[f'A{Gg}',f'V{Gg}',f'S{Gg}',f'B{G}']]
                AA,VV,SS,BB = self.df.loc[0,[f'A{Gg}',f'V{Gg}',f'S{Gg}',f'B{G}']]
            model = gauss(self.xs,AA,VV,SS)+BB
            ax.plot(self.xs, model, label=r'$\sigma$={:.1f}'.format(SS))
        
        model_totl = np.sum([gauss(self.xs,self.df[f'A{G}{g}'].item(),self.df[f'V{G}{g}'].item(),self.df[f'S{G}{g}'].item()) for g in range(1,G+1)],axis=0)+self.df[f'B{G}'].item()
        ax.plot(self.xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
        ax.legend()
        
        if G==1:
            residual = self.y - (gauss(self.x,self.df[f'A{G}'].item(),self.df[f'V{G}'].item(),self.df[f'S{G}'].item()) + self.df[f'B{G}'].item())
        else:
            residual = self.y - (np.sum([gauss(self.x,self.df[f'A{G}{g}'].item(),self.df[f'V{G}{g}'].item(),self.df[f'S{G}{g}'].item()) for g in range(1,G+1)],axis=0)+self.df[f'B{G}'].item())
        
        #plot G2
        ax = axs[1]
        ax.axhline(0, color='gray', alpha=0.5)
        ax.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
        ax.scatter(self.x, residual, s=3, color='tab:blue')
        
        fig.savefig(self.path_plot / "Plotfit_{}GFIT{}_{}.png".format(G,self.suffix,self.name_cube))
    
    # @profile
    def makeplot_corner_emcee(self, savefig: bool = True) -> plt.Figure:
        
        flat_samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)
        flat_physical = map_params(flat_samples, self.gmodel, mode='u->x')
        
        df = pd.DataFrame(flat_physical, columns=self.names_param)

        c = ChainConsumer()
        c.add_chain(Chain(samples=df, name="emcee"))
        
        # truths (dict OK)
        truth_emcee = {p: float(np.ravel(self.df[p].values)[0]) for p in self.names_param}
        c.add_truth(Truth(location=truth_emcee, name="emcee", linestyle=":"))

        fig = c.plotter.plot(figsize=(10,10))
        # self._annotate_spearman_on_chainconsumer_corner(fig, df, self.names_param)
        
        self._annotate_corr_on_chainconsumer_corner(fig,self.df,self.names_param)

        if savefig:
            fig.savefig(self.savename_corner_emcee, transparent=True, bbox_inches="tight")
        return fig


    def makeplot_corner_resample(self, savefig: bool = True) -> plt.Figure:
        names = self.names_param
        resampled = self.resampled
        if 'S21' in names: G=2
        if 'S31' in names: G=3
        if G==2:
            if 'B2' not in names:
                resampled = resampled[:,:-1]
        if G==3:
            if 'B3' not in names:
                resampled = resampled[:,:-1]
        
        df = pd.DataFrame(resampled, columns=names)

        c = ChainConsumer()
        c.add_chain(Chain(samples=df, name="Resampled"))

        # truths (dict OK)
        truth_emcee = {p: float(np.ravel(self.df[p].values)[0]) for p in names}
        truth_resam = {
            p: float(np.nanmedian(np.asarray(sort_outliers(resampled[:, i])).ravel()))
            for i, p in enumerate(names)
        }

        c.add_truth(Truth(location=truth_emcee, name="emcee", linestyle=":"))
        # c.add_truth(Truth(location=truth_resam, name="resampled median", linestyle="--"))

        fig = c.plotter.plot(figsize=(10,10))
        # self._annotate_spearman_on_chainconsumer_corner(fig, df, self.names_param)
        self._annotate_corr_on_chainconsumer_corner(fig,self.df_params,names)

        if savefig:
            fig.savefig(self.savename_corner_resample, transparent=True, bbox_inches="tight")
        return fig
    
    # @profile
    def makeplot_walks(self, axes) -> plt.Figure:
        
        logx = False
        ylim_pad = 0.05
        alpha = 0.05
        lw = 0.6
        
        
        # if ~np.isfinite(self.df_params['e_sn'].item()): return
        # if ~np.isfinite(self.df['SNR2'].item()): return
        
        # df = pd.DataFrame({label: self.resampled[:, i] for i, label in enumerate(self.names_param)})
        # df = pd.DataFrame(self.resampled, columns=self.names_param)

        # consumer = ChainConsumer()
        # consumer.add_chain(Chain(samples=df, name='Resampled'))
        
        
        
        # flat_samples = self.flat_samples
        # df = pd.DataFrame(flat_samples, columns=self.names_param)

        # c = ChainConsumer()
        # c.add_chain(Chain(samples=df, name="emcee"))
        
        # # truths (dict OK)
        # truth_emcee = {p: float(np.ravel(self.df[p].values)[0]) for p in self.names_param}
        # c.add_truth(Truth(location=truth_emcee, name="emcee", linestyle=":"))

        # try:
        #     fig = c.plotter.plot_walks(convolve=100, plot_weights=False, figsize=(6,8))
        # except:
        #     return
        
        if self.sampler is None: return
        if self.gmodel  is None: return
        
        raw = self.sampler.get_chain(discard=0,thin=1)
        nstep_eff,nwalker,ndim = raw.shape
        demap = map_params(raw.reshape(-1,ndim),self.gmodel, mode='u->x').reshape(nstep_eff,nwalker,ndim)
        del raw
        
        names = list(self.gmodel.names_param)
        names_to_i = {n: i for i, n in enumerate(names)}
        
        A_names = [param for param in names if param[0]=='F']
        V_names = [param for param in names if param[0]=='V']
        S_names = [param for param in names if param[0]=='S']
        
        if logx: x = np.arange(1, nstep_eff+1)
        else   : x = np.arange(nstep_eff)
        
        def _shared_ylim(param_names, *, pct=90.0):
            idxs = [names_to_i[p] for p in param_names]
            vals = demap[:, :, idxs].reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return -1.0, 1.0
            lo = (100.0 - pct) / 2.0
            hi = 100.0 - lo
            vmin, vmax = np.nanpercentile(vals, [lo, hi])
            dv = vmax - vmin
            if (not np.isfinite(dv)) or dv == 0:
                return vmin - 1.0, vmax + 1.0
            return vmin - ylim_pad * dv, vmax + ylim_pad * dv

        # -------------------------
        # Plot layout
        # -------------------------
        
        axA,axV,axS = axes

        for pname, color in zip(A_names, ("tab:blue", "tab:orange", "tab:green")):
            j = names_to_i[pname]
            axA.plot(x, demap[:, :, j], color=color, alpha=alpha, linewidth=lw, rasterized=True)
            
        for pname, color in zip(V_names, ("tab:blue", "tab:orange", "tab:green")):
            j = names_to_i[pname]
            axV.plot(x, demap[:, :, j], color=color, alpha=alpha, linewidth=lw, rasterized=True)
            
        for pname, color in zip(S_names, ("tab:blue", "tab:orange", "tab:green")):
            j = names_to_i[pname]
            axS.plot(x, demap[:, :, j], color=color, alpha=alpha, linewidth=lw, rasterized=True)
            
        axA.set_ylabel("Fluxs (demapped)")
        axV.set_ylabel("Velos (demapped)")
        axS.set_ylabel("Disps (demapped)")
        
        y0A, y1A = _shared_ylim(A_names)
        y0V, y1V = _shared_ylim(V_names)
        y0S, y1S = _shared_ylim(S_names)
        axA.set_ylim(y0A, y1A)
        axV.set_ylim(y0V, y1V)
        axS.set_ylim(y0S, y1S)
        axA.grid(True, alpha=0.2)
        axV.grid(True, alpha=0.2)
        axS.grid(True, alpha=0.2)
        
        burnin = self.burnin
        thin   = self.thin
        
        # burn-in marker (note: burnin is in ORIGINAL steps; your displayed x is after discard/thin)
        if burnin is not None and burnin > 0:
            axA.axvline(burnin, color="red", linestyle="--", alpha=0.7)
            axV.axvline(burnin, color="red", linestyle="--", alpha=0.7)
            axS.axvline(burnin, color="red", linestyle="--", alpha=0.7)
            ylimA = axA.get_ylim()
            texty = np.diff(ylimA)*0.05 + ylimA[1]
            axA.text(burnin, texty, f'Burnin\n{burnin}', fontsize=10, color='red')

        # legend
        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="1"),
            Line2D([0], [0], color="tab:orange", lw=2, label="2"),
            Line2D([0], [0], color="tab:green", lw=2, label="3"),
        ]
        axA.legend(handles=legend_handles, loc="upper right", frameon=False)
        axS.legend(handles=legend_handles, loc="upper right", frameon=False)
            
        # delete big arrays explicitly
        del demap
        del axes
                
    def makeplot_GFIT_atlas(self, G, ax_GFIT, ax_Gres, config=None, crop_x=False, crop_x_S1_multiplier=15):
        
        if config=='right':
            ax_GFIT.xaxis.set_tick_params(labelbottom=False)
            ax_GFIT.yaxis.set_tick_params(  labelleft=False)
            ax_Gres.yaxis.set_tick_params(  labelleft=False)
        
        ax = ax_GFIT
        ax.set_title(f'{G}G', fontsize=20)
        ax.xaxis.set_tick_params(labelbottom=False)
        
        ax.axhline(0,color='gray',alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
                
        SNRG = self.df[f'SNR{G}'].item()
        if not np.isfinite(SNRG): return

        for g in range(1,G+1):
            if G==1: AA,VV,SS,BB = self.df.loc[0,[f'A{g}',f'V{g}',f'S{g}',f'B{g}']]
            else:
                Gg = f'{G}{g}'
                AA,VV,SS,BB = self.df.loc[0,[f'A{Gg}',f'V{Gg}',f'S{Gg}',f'B{G}']]
            model = gauss(self.xs,AA,VV,SS)+BB
            ax.plot(self.xs, model, label=r'$\sigma$={:.1f}'.format(SS))
        
        if G==1:
            model_totl = gauss(self.xs,self.df[f'A{G}'].item(),self.df[f'V{G}'].item(),self.df[f'S{G}'].item()) + self.df[f'B{G}'].item()
        else:
            model_totl = np.sum([gauss(self.xs,self.df[f'A{G}{g}'].item(),self.df[f'V{G}{g}'].item(),self.df[f'S{G}{g}'].item()) for g in range(1,G+1)],axis=0)+self.df[f'B{G}'].item()
        ax.plot(self.xs, model_totl, color='black', alpha=0.5, label=r'$\Sigma$')
        
        BIC = self.df[f'BIC{G}'].item()
        bic_text = f'BIC={BIC:.1f}'
        if G > 1:
            BIC_prev = self.df[f'BIC1'].item()
            dBIC = BIC - BIC_prev
            bic_text += f'\n$\\Delta \\ BIC_{{{G}-1}}={dBIC:.1f}$'
        ax_GFIT.text(0.01, 0.99, bic_text, va='top', ha='left',
                    transform=ax_GFIT.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.legend(title='S/N={:.0f}'.format(SNRG), loc='upper right')

        if G==1:
            model_totl = gauss(self.x,self.df[f'A{G}'].item(),self.df[f'V{G}'].item(),self.df[f'S{G}'].item()) + self.df[f'B{G}'].item()
        else:
            model_totl = np.sum([gauss(self.x,self.df[f'A{G}{g}'].item(),self.df[f'V{G}{g}'].item(),self.df[f'S{G}{g}'].item()) for g in range(1,G+1)],axis=0)+self.df[f'B{G}'].item()
        residuals = self.y - model_totl
    
        #plot G2
        ax = ax_Gres
        ax.axhline(0, color='gray', alpha=0.5)
        ax.set_xlabel(r'$\mathrm{km \ s^{-1}}$')
        ax.scatter(self.x, residuals, s=3, color='tab:blue')

        sigma_res = np.nanstd(residuals)        
        sigma_eff = np.maximum(self.e_y, sigma_res)
        chisq = np.sum((residuals/sigma_eff)**2)
        dof   = len(self.y)-len(self.names_param)
        chisq_red = chisq/dof
        ax_Gres.text(0.99,0.99, r'$\chi^2_\mathrm{red}$='+f'{chisq_red:.1f}', va='top',ha='right', transform=ax_Gres.transAxes)

        # F, crit = self.df.loc[0,['F-test','F-crit']]        
        # ax_2Gres.text(0.01,0.99, r'F='+f'{F:.2f}({crit:.2f})', va='top',ha='left', transform=ax_2Gres.transAxes)
        
        NG,BG = self.df.loc[0,[f'N{G}',f'B{G}']]
        
        # Noise
        ax_GFIT.axhspan(  -NG+BG,   NG+BG, color='gray', alpha=0.2, zorder=0)
        ax_GFIT.axhspan(-3*NG+BG, 3*NG+BG, color='gray', alpha=0.2, zorder=0)
        
        xlim = ax_GFIT.get_xlim()
        ax_GFIT.text(xlim[1], 3*NG+BG, rf'$3 \mathrm{{RMS}}_{{\mathrm{{{G}G}}}}$', va='top', ha='right', color='gray')
        
        if crop_x:
            S1 = self.df['S1'].item()
            ax_GFIT.set_xlim(-crop_x_S1_multiplier*S1,crop_x_S1_multiplier*S1)
        
        ax_Gres.text(0.99,0.01, r'RMS='+f'{NG*1000:.2f} mJy', va='bottom',ha='right', transform=ax_Gres.transAxes)

    def makeplot_GFIT_resampled(self, ax):
        
        if 'S21' in self.names_param: G=2
        if 'S31' in self.names_param: G=3
        
        self.nsample_resample = len(self.resampled[:,0])
        alpha = np.max([1/self.nsample_resample,1/510.])
        
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(  labelleft=False)
        
        ax.axhline(0, color='gray', alpha=0.5)
        ax.errorbar(self.x, self.y, self.e_y, alpha=0.5, color='gray', fmt='.', elinewidth=0.5)
        
        colors = ['tab:blue','tab:orange','tab:green']
        
        segments_by_g = {1: [], 2: [], 3: []}
        
        BBs = self.resampled[:,-1]
        for i in range(self.nsample_resample):
            for g in range(1,G+1):
                Gg = f'{G}{g}'
                
                if np.isin(f'V{Gg}',self.names_param):
                    iV = idx(self.names_param,f'V{Gg}')
                    VV = self.resampled[i,iV]
                else:
                    VV = self.df[f'V{Gg}'].item()
                    
                iS = idx(self.names_param,f'S{Gg}')
                if g==3 and 'S32' not in self.names_param: SS = self.df['S32'].item()
                else:
                    SS = self.resampled[i,iS]
                
                iF = idx(self.names_param,f'F{Gg}')
                FF = self.resampled[i,iF]
                AA = FF / (SS * 2.50662872463)
                
                # iA = idx(self.names_param,f'A{Gg}')
                # AA = self.resampled[i,iA]
                
                model = gauss(self.xs, AA,VV,SS)+BBs[i]
                segments_by_g[g].append(np.column_stack([self.xs, model]))
        
        for g,color in zip(range(1,G+1),colors):
            lc = LineCollection(segments_by_g[g], colors=color, linewidths=0.5, alpha=alpha, zorder=0)
            # ax.plot(self.xs, model, alpha=alpha, lw=0.5, zorder=0, color=colors[g-1])
            ax.add_collection(lc)
        
        
    def makeplot_disphist(self, ax:plt.Axes) -> None:
        
        if len(self.list_disp)<2: return
        weights = self.list_NHI_ if self.list_NHI_ is not None else None
        
        ax.set_box_aspect(1)
        
        # ax.plot(self.kde_x,self.kde_y,color='tab:gray')
        ax.set_ylim(bottom=0)
        
        ax2 = ax.twinx()
        bins = np.arange(0,np.nanmax(self.list_disp),1)
        # sns.histplot(x=self.list_disp, bins=bins, color='tab:gray', alpha=0.5, weights=weights, ax=ax2, edgecolor=None, kde=True)
        
        sns.histplot(x=self.list_disp, binwidth=2, color='tab:gray', alpha=0.5, weights=weights, ax=ax2, edgecolor=None, kde=True)
                
        S1 = self.df['S1'].item()
        ax.axvline(S1, color='black')
        ymax = ax.get_ylim()[1]
        ax.text(S1+0.5,ymax*0.98, r'$\sigma_\mathrm{1G}$'+f'\n{S1:.1f}', ha='left', va='top')
        
        if 'e_S1' in self.df_params and np.isfinite(self.df_params['e_S1'].item()):
            e_S1 = self.df_params['e_S1'].item()
            ax.axvspan(S1 - e_S1, S1 + e_S1, color='black', alpha=0.3, zorder=0)
        
        ax.set_xlim(0,np.max([np.nanpercentile(self.list_disp, 99), S1*3]))
        
        percentiles = [10,20,30,40,50,60,70,80,90,95,96,97,98,99]
        vals_percentile = np.percentile(self.list_disp,percentiles, weights=weights, method='inverted_cdf')
        for i,val in enumerate(vals_percentile):
            ax.plot([val,val],[ymax*0.5,ymax*0.55], color='black')
            ax.text(val, ymax*0.551, f'{percentiles[i]:.0f}', va='bottom', ha='center')
            
        ax.set_xlabel(r'$\mathrm{km \, s^{-1}}$')
        
        G = None
        if np.isin('S21',self.names_param): G=2
        if np.isin('S31',self.names_param): G=3
        if G is None: return
        
        if not np.isfinite(self.df[f'SNR{G}'].item()): return

            # ax.axvspan(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S22'][1], color='lightgray', alpha=0.5)
            
        colors = ['tab:blue','tab:orange','tab:green']
        
        if G==2:
            keys_p = ['S21','S22']
            keys_e = ['sn','sb']
        if G==3:
            keys_p = ['S31','S32','S33']
            keys_e = ['sn','sb','sw']
            
        for i, (key_p, key_e) in enumerate(zip(keys_p,keys_e)):
            SS = self.df[key_p].item()
            ax.axvline(SS, color=colors[i])
            ax.text(   SS, ymax*1.01, key_p+f'\n{SS:.2f}', ha='right', va='bottom')
            if key_e in self.df_params and np.isfinite(self.df_params[key_e].item()):
                # e_SS = self.df_params[key_e].item()
                e_lower = self.df_params['e-_'+key_e].item()
                e_upper = self.df_params['e+_'+key_e].item()
                
                ax.axvspan(SS - e_lower, SS + e_upper, color=colors[i], alpha=0.3, zorder=0)
            
        # ax.text(ax.get_xlim()[1],ymax*1.01, r'$\sigma_\mathrm{b}-\sigma_\mathrm{n}$'+'\n{:.2f}'.format(S22-S21), ha='right', va='bottom')
            
            # text  = 'Bounds'
            # text += '\nS21=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S21'][1])
            # text += '\nS22=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S22'][0],self.gmodel.dict_bound['S22'][1])
            
            # ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
        # else:
        #     text  = 'Bounds'
        #     text += '\nS1=({:.2f},{:.2f})'.format(self.gmodel.dict_bound['S21'][0],self.gmodel.dict_bound['S21'][1])
        #     ax.text(0.99,0.01, text,ha='right',va='bottom', transform=ax.transAxes)
            
        
        
    # def makeplot_paramshist(self, key, savefig:bool=True, transparent:bool=True) -> None:
        
    #     if 'A31' in self.names_param: G = 3
    #     if 'A21' in self.names_param: G = 2
        
    #     names = self.names_param
        
    #     iSX1 = idx(names, f'S{G}1')
    #     iSX2 = idx(names, f'S{G}2')
            
    #     sn = self.resampled[:,np.argwhere(self.names_param==f'S{G}1').item()]
    #     sb = self.resampled[:,np.argwhere(self.names_param==f'S{G}2').item()]
    #     An = gaussian_area(self.resampled[:,np.argwhere(self.names_param==f'A{G}1').item()], sn)
    #     Ab = gaussian_area(self.resampled[:,np.argwhere(self.names_param==f'A{G}2').item()], sb)
    #     At = An+Ab
        
    #     dict_resampled = {
    #         'sn':sn,
    #         'sb':sb,
    #         'An':An,
    #         'Ab':Ab,
    #         'At':At,
            
    #         'sn/sb':sn/sb,
    #         'An/At':An/At,
    #         'log(sb-sn)':np.log10(sb-sn),
    #     }
        
    #     for keyy in dict_resampled.keys():
    #         dict_resampled[keyy] = sort_outliers(dict_resampled[keyy])
        
    #     # dict_resampled['sn/sb'] = dict_resampled['sn/sb'][dict_resampled['sn/sb']<0.9]
    #     # print(dict_resampled)
        
    #     dict_title = {
    #         'sn':r'$\sigma_\mathrm{n}$',
    #         'sb':r'$\sigma_\mathrm{b}$',
    #         'An':r'$A_\mathrm{n}$',
    #         'Ab':r'$A_\mathrm{b}$',
    #         'At':r'$A_\mathrm{tot}$',
            
    #         'sn/sb':r'$\sigma_\mathrm{n}/\sigma_\mathrm{b}$',
    #         'An/At':r'$A_\mathrm{n}/A_\mathrm{tot}$',
    #         'log(sb-sn)':r'$\log (\sigma_\mathrm{b}-\sigma_\mathrm{n})$',
    #     }
        
    #     dict_xlim = {
    #         'sn/sb':[0,1],
    #         'An/At':[0,1],
    #         'log(sb-sn)':[0,2]
    #     }
                
    #     df = pd.DataFrame(dict_resampled, columns=self.names_param)

    #     print(key)
    #     consumer = ChainConsumer()
    #     consumer.add_chain(
    #         df.to_numpy(),                 # shape (n_samples, n_params)
    #         parameters=df.columns.tolist(),# parameter names
    #         name="Resampled"
    #     )
        
    #     # consumer.add_truth(Truth(location={key:self.df_params[key].item()}, name='emcee', color='tab:blue'))
    #     # consumer.add_truth(Truth(location={key:np.median(dict_resampled[key])}, name='resampled', color='tab:orange'))
        
    #     try:
    #         fig = consumer.plotter.plot_distributions(figsize=(4,5))#columns=[dict_title[key]])
    #     except Exception as e:
    #         print(e)
    #         return
        
    #     if savefig:
    #         ax = fig.get_axes()[0]
    #         ax.set_box_aspect(1)
            
    #         ax.axvline(self.df_params[key].item(), color='tab:blue')
    #         ax.axvline(np.mean(dict_resampled[key]), color='tab:orange')
            
    #         ax.axvspan(self.df_params[key].item()-self.df_params['e_'+key].item(),
    #                    self.df_params[key].item()+self.df_params['e_'+key].item(),
    #                    color='tab:blue', alpha=0.1, label='emcee', zorder=0)
    #         ax.axvspan(np.mean(dict_resampled[key])-np.std(dict_resampled[key]),
    #                    np.mean(dict_resampled[key])+np.std(dict_resampled[key]),
    #                    color='tab:orange', alpha=0.1, label='resampled', zorder=0)
            
    #         ax.set_xlim(dict_xlim[key][0],dict_xlim[key][1])
            
    #         fig.savefig(self.savename_paramshist, transparent=True)
        
    #     return fig
    def makeplot_atlas(self, G=2) -> None:
        
        def dict_coord_to_subplot_coord(dict_coord):
            return [dict_coord['l'],dict_coord['b'],dict_coord['r']-dict_coord['l'],dict_coord['t']-dict_coord['b']]
        
        def paste_image(fig, path_image, dict_coord):
            if not os.path.exists(path_image):
                return

            # frontImage = Image.open(path_image)
            # width_pix = int(dict_coord['r']-dict_coord['l'])
            # heigt_pix = int(dict_coord['t']-dict_coord['b'])
            # frontImage_red = frontImage.resize((width_pix,heigt_pix))
            # background.paste(frontImage_red, [int(dict_coord['l']), int(bgheight-dict_coord['t'])], frontImage_red)
            
            ax_img = fig.add_axes(dict_coord_to_subplot_coord(dict_coord))
            img = plt.imread(path_image)
            ax_img.imshow(img)
            ax_img.axis('off')
        
        suffix = self.suffix if self.suffix!='' else '_'
        

        fig = plt.figure(figsize=(25,15))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        
        # buf = io.BytesIO()
        # fig.savefig(buf, format="png")
        # plt.close(fig)
        # buf.seek(0)
        # background = Image.open(buf)
        
        # bgwidth, bgheight = background.size
        
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
        ax_1GFIT.set_ylabel('Jy', fontsize=12)
        
        coord_frame_XGFIT = {'t':coord_frame_1GFIT['t'],
                    'l':coord_frame_1GFIT['r']+0.01,       'r':coord_frame_1GFIT['r']*2-coord_frame_1GFIT['l']+0.01,
                             'b':coord_frame_1GFIT['b'] }
        coord_frame_XGres = {'t':coord_frame_XGFIT['b']-0.01,
                    'l':coord_frame_XGFIT['l'],       'r':coord_frame_XGFIT['r'],
                             'b':coord_frame_1Gres['b'] }
        ax_XGFIT = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_XGFIT), sharex=ax_1GFIT, sharey=ax_1GFIT)
        ax_XGres = fig.add_subplot(dict_coord_to_subplot_coord(coord_frame_XGres), sharex=ax_1GFIT, sharey=ax_1Gres)
        
        self.makeplot_GFIT_atlas(G, ax_XGFIT, ax_XGres, config='right')
        
        # if G==2 and 'F21' in self.df and np.isfinite(self.df.loc[0,'F21']):
        #     F21,F22 = self.df.loc[0,[f'F{G}1',f'F{G}2']]
        #     txt = f'F21={F21:.2f}\nF22={F22:.2f}'
        #     ax_2GFIT.text(0.99,0.5, txt, ha='right', va='center', transform=ax_2GFIT.transAxes)
        
        # if G==3 and 'F31' in self.df and np.isfinite(self.df.loc[0,'F31']):
        #     F31,F32,F33 = self.df.loc[0,[f'F{G}1',f'F{G}2',f'F{G}3']]
        #     txt = f'F31={F31:.2f}\nF32={F32:.2f}\nF33={F33:.2f}'
        #     ax_2GFIT.text(0.99,0.5, txt, ha='right', va='center', transform=ax_2GFIT.transAxes)
        
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
        height_panels_walk = 0.125
        coord_frame_walks_S = {'t':0.07+height_panels_walk,
                               'l':coord_disphist['l'], 'r':0.25,
                               'b':0.07}
        coord_frame_walks_V = {'t':coord_frame_walks_S['t']+0.01+height_panels_walk,
                               'l':coord_disphist['l'], 'r':0.25,
                               'b':coord_frame_walks_S['t']+0.01}
        coord_frame_walks_F = {'t':coord_frame_walks_V['t']+0.01+height_panels_walk,
                               'l':coord_disphist['l'], 'r':0.25,
                               'b':coord_frame_walks_V['t']+0.01}
        
        coord = coord_frame_walks_S
        ax_walks_S = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']])
        coord = coord_frame_walks_V
        ax_walks_V = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']], sharex=ax_walks_S)
        coord = coord_frame_walks_F
        ax_walks_F = fig.add_subplot([coord['l'],coord['b'],coord['r']-coord['l'],coord['t']-coord['b']], sharex=ax_walks_S)
        
        plt.setp(ax_walks_F.get_xticklabels(),visible=False)
        plt.setp(ax_walks_V.get_xticklabels(),visible=False)

        if self.sampler is not None:
            self.makeplot_walks((ax_walks_F,ax_walks_V,ax_walks_S))

        
        #===============
        
        coord_corner_emcee = {
                'l':0.27,
                'r':0.56,
                't':0.95,
                'b':0.51,
            }
        
        coord_corner_resample = {
                'l':coord_corner_emcee['l'],
                'r':coord_corner_emcee['r'],
                't':0.48,
                'b':0.04,
            }
        
        coord_GFIT_resample = {
                'l':0.60,
                'r':0.70,
                't':0.48,
                'b':0.32,
            }
        
        if 'e-_sn' in self.df_params and np.isfinite(self.df_params['e-_sn'].item()):
            ax_GFIT_resamp = fig.add_subplot(dict_coord_to_subplot_coord(coord_GFIT_resample))
            self.makeplot_GFIT_resampled(ax_GFIT_resamp)
            
        # buf = io.BytesIO()
        # fig.savefig(buf, format="png")
        # plt.close(fig)
        # buf.seek(0)
        # background = Image.open(buf)
        
        # background = Image.open(self.savename_atlas)
        # bgwidth, bgheight = background.size
        
        # if(np.isfinite(self.df[f'SNR{G}'].item())):
        # if f'e+_A{G}1' in self.df:
        #     if (np.isfinite(self.df[f'e+_A{G}1'].item())) :
        
        if self.sampler is not None and G>1:
            if self.savename_corner_emcee.exists()==False:
                try:
                    self.makeplot_corner_emcee(savefig=True)
                except IndexError:
                    pass
            if self.savename_corner_emcee.exists():
                paste_image(fig, self.savename_corner_emcee, coord_corner_emcee)
        
        if self.resampled is not None and G>1:
            if self.savename_corner_resample.exists()==False:
                try:
                    self.makeplot_corner_resample(savefig=True)
                except IndexError:
                    pass
            if self.savename_corner_resample.exists():
                paste_image(fig, self.savename_corner_resample, coord_corner_resample)
                        
        # if 'sn' in self.df_params:
        #     keys = ['sn/sb','An/At','log(sb-sn)']
        #     width = (coord_GFIT_resample['r']-coord_GFIT_resample['l'])+0.015
        #     height = 0.17
        #     offset = 0.007
        #     left = (coord_GFIT_resample['l'] - 0.01)*bgwidth
        #     for i, key in enumerate(keys):
        #         coord_paramshist = {
        #                 'l': left,
        #                 'r': left + (width * bgwidth),
        #                 't':(0.105+height)*bgheight,
        #                 'b':(0.06)*bgheight,
        #         }
        #         left = coord_paramshist['r']+(offset*bgwidth)
        #         if np.isfinite(self.df_params['e_sn'].item()):
        #             self.makeplot_paramshist(key=key)
        #             try: paste_image(self.savename_paramshist, coord_paramshist)
        #             except AttributeError or FileNotFoundError: pass
                    
        # background.save(self.savename_atlas, format="png")
        fig.savefig(self.savename_atlas, format="png", dpi=80)
        
        gc.collect()
        del fig
        
        
        return