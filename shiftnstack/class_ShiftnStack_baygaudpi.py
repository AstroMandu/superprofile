import glob
import os
# from .class_Filter_genuine import Filter
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from natsort import natsorted
from radio_beam.beam import Beam
from routines_baygaudpi import gauss, read_ngfits
from run_makemask_ellipse import run_makemask as run_makemask_ellipse
from spectral_cube import SpectralCube
from spectral_cube.io.core import StokesWarning
from tqdm import tqdm

from .subroutines_shiftnstack import (argfind_nearest, shifter_gipsy,
                                      shifter_roll)

# matplotlib.use('Agg')

warnings.simplefilter("ignore", StokesWarning)


class ShiftnStack:
    
    def __init__(self, path_cube, path_classified, path_mask=None, bgsub=True, path_vf_secondary=None, correct_vf_secondary=False):
        path_cube       = Path(path_cube)
        path_classified = Path(path_classified)
        
        self.path_cube = path_cube
        self.path_clfy = path_classified
        self.path_mask = path_mask
        
        self.name_cube = path_cube.parent.name
        
        hedr_cube = fits.getheader(path_cube)
        data_cube = fits.getdata(path_cube) #* (u.Jy/u.beam)
        if(len(data_cube.shape)>3): data_cube = data_cube[0,:,:,:]
        
        spec_axis = (SpectralCube.read(path_cube)).with_spectral_unit(u.km/u.s, velocity_convention='optical').spectral_axis.value
        chansep = np.diff(spec_axis)[0]
        
        self.hedr_cube = hedr_cube
        self.data_cube = data_cube 
        
        if path_vf_secondary is not None:
            path_vf_secondary = Path(path_vf_secondary)
            data_vf_secondary = (fits.getdata(path_vf_secondary)*(u.m/u.s)).to(u.km/u.s).value
            if len(data_vf_secondary.shape)>2: data_vf_secondary = data_vf_secondary[0,:,:]
            data_vf_secondary = data_vf_secondary
            
            if correct_vf_secondary:
                # radio velocity -> optical velocity
                import astropy.constants as const
                c = const.c.to('km/s').value
                data_vf_secondary = c * (1 / (1 - data_vf_secondary / c) - 1)
            
        else:
            data_vf_secondary = np.full((hedr_cube['NAXIS2'],hedr_cube['NAXIS1']),np.nan)
        self.data_vf_secondary = data_vf_secondary
        
        if(path_mask is not None):
            path_mask = Path(path_mask)
            data_mask = fits.getdata(path_mask)
            if(len(data_mask.shape)>2):
                data_mask = data_mask[0,:,:]
            # data_cube = np.where(np.isfinite(np.broadcast_to(data_mask, data_cube.shape)), data_cube, np.nan)
        else:
            data_mask = np.ones((hedr_cube['NAXIS2'],hedr_cube['NAXIS1']))
        self.data_mask = data_mask
        
        n = data_cube.shape[0]
        n_edge = max(n // 10, 10)  # at least 10 channels each side

        edge_channels = np.concatenate([
            data_cube[:n_edge, :, :],
            data_cube[-n_edge:, :, :]
        ], axis=0)  # shape (2*n_edge, ny, nx)

        std_channel = mad_std(edge_channels, ignore_nan=True)
        self.std_channel = std_channel * u.Jy/u.beam
           
        nopt = len(glob.glob(str(path_classified/"ngfit/*G*_*.0.fits")))
        
        map_ngau = fits.getdata(glob.glob(str(path_classified/"sgfit/*.7.fits"))[0])
        # 
        
        self.len_nopt = int(np.nansum(map_ngau))
        if bgsub:
            map_bkgr = fits.getdata(glob.glob(str(path_classified/"ngfit/*_1.3.fits"))[0])#*(u.Jy/u.beam)
            self.data_cube-= np.where(np.isfinite(map_bkgr),map_bkgr,0)
        
        self.spec_axis = spec_axis
        self.chansep    = chansep
        self.abschansep = np.abs(chansep)
        self.nopt = nopt
        
        beam = Beam.from_fits_header(hedr_cube)
        self.beam = beam
        self.bM = beam.major.to(u.arcsec)
        self.bm = beam.minor.to(u.arcsec)
        self.cd = (hedr_cube['CDELT2']*u.degree).to(u.arcsec)
        self.len_sa = len(self.spec_axis)
        
        self.area_beam = self.beam.sr
        self.area_pixl = (self.cd**2).to(u.sr)
        self.pix_per_beam = self.area_beam/self.area_pixl
        
        self.dict_stacked = {}
        
        self.pbar = False
        self.bgsub = bgsub
        self.stat  = 'STBY'
        self.path_temp = None
        
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
        
    def _mom0_to_NHI(self, mom0):
        NHI = (
            (mom0 * u.Jy).to(u.K, u.brightness_temperature(1.420 * u.GHz, self.beam)).value
            * 1e-3 * 1.823e18
        )
        return NHI
        
        
    def mask_specrange(self, dict_mask):
        """
        Mask (zero out) a range along the spectral axis of the cube.

        Parameters
        ----------
        low : astropy.units.Quantity | float | int | None
            Lower bound of the mask. Meaning depends on `mode`.
            - mode='velocity': km/s quantity (or numeric already in km/s)
            - mode='channel' : channel index (int, can be negative)
            If None, mask from the start up to `high`.
        high : astropy.units.Quantity | float | int | None
            Upper bound of the mask. Same unit/index semantics as `low`.
            If None, mask from `low` to the end.
        mode : {'velocity', 'channel'}
            How to interpret `low`/`high`.
            - 'velocity': compare against `self.spec_axis` (km/s)
            - 'channel' : compare against channel indices [0..N-1]

        Notes
        -----
        - Boundaries are *open* (low < x < high), matching your original code.
        - The spectral axis is assumed to be axis 0.
        - The selected range is set to 0 (masked); everything else is kept.
        """
        import astropy.units as u
        import numpy as np

        n_spec = self.data_cube.shape[0]
        
        fr   = dict_mask['from']
        to   = dict_mask['to']
        mask = dict_mask['mask']
        mode = dict_mask['mode']
        
        if mode == 'velocity':
            pass
            # # Ensure we compare in km/s
            # spec = np.asarray(self.spec_axis)  # already numeric (km/s) in your code
            # def to_kms(x):
            #     if x is None:
            #         return None
            #     return x.to(u.km/u.s).value if hasattr(x, 'to') else float(x)

            # low_v  = to_kms(low)
            # high_v = to_kms(high)

            # if low_v is None and high_v is None:
            #     mask = np.zeros_like(spec, dtype=bool)  # nothing to mask
            # elif low_v is None:
            #     mask = spec < high_v
            # elif high_v is None:
            #     mask = spec > low_v
            # else:
            #     if high_v < low_v:  # tolerate swapped inputs
            #         low_v, high_v = high_v, low_v
            #     mask = (spec > low_v) & (spec < high_v)

        elif mode == 'channel':
            chans = np.arange(n_spec)
            if mask=='inner':
                mask = (chans > fr) & (chans < to)
            elif mask=='outer':
                mask = (chans < fr) | (chans > to)
        else:
            raise ValueError("mode must be 'velocity' or 'channel'")

        # Apply along spectral axis (assumed axis 0)
        mask_reshaped  = mask[:, np.newaxis, np.newaxis]
        self.data_cube = np.where(mask_reshaped, 0, self.data_cube)
        
    def filter_genuine(self, dict_G):
        
        nois   = dict_G['nois']
        velo   = dict_G['velo']
        bkgr   = dict_G['bkgr']
        disp   = dict_G['disp']
        e_disp = dict_G['e_disp']
        
        if disp<2*e_disp: return False
        
        model =  gauss(self.spec_axis, dict_G['ampl'], velo, disp)
        mask = (self.spec_axis>velo-3*disp)&(self.spec_axis<velo+3*disp)&\
               (model>2*nois+bkgr)

        if len(mask)>3:
            return True
        
        else: 
            return False
    
    def run(self, stack_secondary=False):
        
        # def argfind_nearest(array, value):
        #     idx = np.searchsorted(array, value)
        #     if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        #         return idx-1
        #     else:
        #         return idx
       
        data_cube = np.float64(self.data_cube.copy())
        dict_data = read_ngfits(self.path_clfy/'ngfit',
                                toreads=['flux','velo','disp','psnr','bkgr','nois','e_disp'], 
                                path_mask=self.path_mask,
                                wo_unit=True)
        
        if self.bgsub:
            for coord in dict_data.keys():
                for g in dict_data[coord].keys():
                    dict_data[coord][g]['bkgr'] = 0.
        
        data_mask = self.data_mask.copy()
        data_vf_secondary = self.data_vf_secondary
        
        for coord in dict_data.keys():
            for g in dict_data[coord].keys():
                
                if 'psnr' not in dict_data[coord][g]: continue
            
                psnr = dict_data[coord][g]['psnr']
                nois = dict_data[coord][g]['nois']
                bkgr = dict_data[coord][g]['bkgr']
                ampl = (psnr * nois) + bkgr
                # ampl = (psnr * nois)
                
                dict_data[coord][g]['ampl'] = ampl
        
        # ROLL ---------------------------------------------------
        len_specaxis = len(self.spec_axis)
        xx = np.arange(len_specaxis) - int((len_specaxis - 1)/2)
        if self.chansep < 0:
            xx = xx[::-1]
        lenx = len(xx)
        # shifter = shifter_roll
        shifter = shifter_gipsy
        # --------------------------------------------------------
        
        # CUT ----------------------------------------------------
        # len_specaxis = len(self.spec_axis)
        # if self.chansep<0:
        #     xx = np.arange(len_specaxis-1, -len_specaxis, -1)
        # else:
        #     xx  = np.arange(-len_specaxis+1, len_specaxis, 1)
        # lenx = len(xx)
        # --------------------------------------------------------
            
        yy  = np.zeros_like(xx, dtype=np.float64)#*(u.Jy/u.beam)
        e_y = np.zeros_like(xx, dtype=np.float64)#*(u.Jy/u.beam)
        NN  = np.zeros_like(xx)
        
        sa_div_chan = self.spec_axis / self.chansep
        index_centr = np.argwhere(xx==0).item()
        
        std_channel = self.std_channel
            
        # data_cube[np.isnan(data_cube)] = np.random.normal(0,std_channel.value)#*data_cube.unit
    
        # from tqdm import tqdm
        # for coord in tqdm(dict_data.keys()):
        
        len_data = len(dict_data)
                
        if self.pbar: pbar = tqdm(total=len_data)
        index_dict_stacked = 0
        count = 0
        last_reported = -1
        
        self.stat = 'STAKB'
        for coord in dict_data.keys():
            x,y = map(int, coord.split(','))
            dict_cord = dict_data[coord]
            
            data_mask[y,x] = np.nan
            
            all_models = {
                g: gauss(self.spec_axis, d['ampl'], d['velo'], d['disp'])
                for g, d in dict_cord.items()
                if 'velo' in d
            }
            model_sum = sum(all_models.values()) if all_models else 0.0
            
            for g in dict_cord:
                if 'velo' not in dict_cord[g]: continue
                
                dict_pg = dict_cord[g]
                
                data_subed  = data_cube[:,y,x].copy()
                data_subed -= (model_sum - all_models[g])
                data_subed -= dict_pg['bkgr']

                # shift = dict_pg['velo']/self.chansep
                # index = argfind_nearest(sa_div_chan, shift)

                # mod = index_centr - index
                # shifted = shifter(data_subed, mod, lenx)
                # yy += shifted
                # NN += (shifted != 0).astype(np.uint8)
                
                shifted, valid = shifter(
                    data_subed,          # spectrum
                    self.spec_axis,      # world coord per channel (km/s optical in your pipeline)
                    dict_pg['velo'],     # center velocity
                    xx*self.abschansep,                  # your offset bins in channel units
                    fill=0.0,            # keep 0 outside range to match your current NN logic style
                    index_centr = index_centr
                )

                yy += shifted
                NN += valid.astype(np.uint8)
                                
                # self.dict_stacked[count] = {'disp':dict_pg['disp'], 'e_disp':dict_pg['e_disp']}
                self.dict_stacked[index_dict_stacked] = {
                    'NHI':self._mom0_to_NHI(dict_pg['flux'])*1000,
                    'disp':dict_pg['disp'], 'e_disp':dict_pg['e_disp']}
                index_dict_stacked+=1
                
                # if mod<0: 
                #     showplot=True
                    
                # if showplot:
                #     import pylab as plt
                #     fig, axs = plt.subplots(nrows=4)
                #     plt.rcParams['hatch.linewidth']=4
                #     ax=axs[0]
                #     ax.axhline(0,color='gray',alpha=0.5)
                #     ax.step(self.spec_axis, data_cube[:,y,x],color='gray',where='mid')
                #     ax.fill_between(self.spec_axis, data_cube[:,y,x], step='mid', hatch=r"//", color='lightgrey', edgecolor='white')

                #     fig.suptitle("({}, {}), Nopt={}".format(x, y, len(dict_cord)))
                #     ax.set_xlabel(r"Velocity [km $s^{-1}]$", color='white', fontsize=20, labelpad=-3)
                #     ax.set_ylabel(r"Intensity [mJy/beam]",   color='white', fontsize=20)
                    
                #     xs = np.linspace(self.spec_axis.min(),self.spec_axis.max(),100)
                    
                #     model_tot = np.zeros_like(xs)
                #     for gother in dict_cord.keys():
                #         if(g==gother): 
                #             model = gauss(xs, dict_pg['ampl'], dict_pg['velo'], dict_pg['disp'])#+dict_pg['bkgr']
                #             ax.plot(xs,model,color='tab:blue')
                #             model_tot+=model#-dict_pg['bkgr']
                #             continue
                #         dict_pgother = dict_cord[gother]
                #         model = gauss(xs, dict_pgother['ampl'], dict_pgother['velo'], dict_pgother['disp'])#+dict_pg['bkgr']
                #         ax.plot(xs,model,color='gray')
                #         model_tot+=model
                #     ax.plot(xs,model_tot, color='black',alpha=0.5)
                    
                #     ax=axs[1]
                #     ax.axhline(0,color='gray',alpha=0.5)
                #     ax.step(        self.spec_axis, data_subed,color='gray',where='mid')
                #     ax.fill_between(self.spec_axis, data_subed, step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                #     ax.set_ylim(axs[0].get_ylim())

                #     fig.suptitle("({}, {}), Nopt={}".format(x, y, len(dict_cord)))
                #     ax.set_xlabel(r"Velocity [km $s^{-1}]$", color='white', fontsize=20, labelpad=-3)
                #     ax.set_ylabel(r"Intensity [mJy/beam]",   color='white', fontsize=20)
                    
                    
                #     ax=axs[2]
                #     ax.axhline(0,color='gray',alpha=0.5)
                #     ax.step(        xx*np.abs(self.chansep), shifted, where='mid', color='gray')
                #     ax.fill_between(xx*np.abs(self.chansep), shifted, step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                #     ax.set_ylim(axs[0].get_ylim())
                    
                #     ax=axs[3]
                #     ax.axhline(0,color='gray',alpha=0.5)
                #     ax.step(        xx*np.abs(self.chansep), yy, where='mid', color='gray')
                #     ax.fill_between(xx*np.abs(self.chansep), yy, step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                    
                #     plt.savefig(f'/home/mskim/workspace/research/temp/{count}.png')
                #     # plt.close(fig)
                    
            if self.pbar: pbar.update(1)
            count+=1
            prog_perc = count/len_data*100
            if prog_perc // 10 > last_reported:
                last_reported = prog_perc // 10
                self.writestat(f"{self.stat} {prog_perc:.0f}% .")
        
        if stack_secondary:
            
            if os.path.exists(self.path_cube.parent/'cube_mom2.fits'):
                data_mom0 = fits.getdata(self.path_cube.parent/'cube_mom0.fits')
                data_mom2 = fits.getdata(self.path_cube.parent/'cube_mom2.fits')/1000.
            else: data_mom2 = None
            
            data_vf_secondary = np.where(data_mom2<self.chansep/2.355*2, np.nan, data_vf_secondary)
            
            count_plot = 0
            count = 0
            last_reported = -1
            len_data = len(np.argwhere((np.isfinite(data_mask)) & (np.isfinite(data_vf_secondary))))
            self.stat = 'STAKM'
            for y,x in np.argwhere((np.isfinite(data_mask)) & (np.isfinite(data_vf_secondary))):
                # shift = data_vf_secondary[y,x]/self.chansep
                # index = argfind_nearest(sa_div_chan, shift)
                # mod = index_centr - index
                # shifted = shifter(data_cube[:,y,x], mod, lenx)
                # yy  += shifted
                # NN  += (shifted != 0).astype(np.uint8)
                
                shifted, valid = shifter(
                    data_cube[:,y,x],          # spectrum
                    self.spec_axis,      # world coord per channel (km/s optical in your pipeline)
                    data_vf_secondary[y,x],     # center velocity
                    xx*self.abschansep,                  # your offset bins in channel units
                    fill=0.0,            # keep 0 outside range to match your current NN logic style
                    index_centr = index_centr
                )
                yy += shifted
                NN += valid.astype(np.uint8)
                
                if data_mom2 is not None:
                    self.dict_stacked[index_dict_stacked] = {
                        'NHI' :self._mom0_to_NHI(data_mom0[y,x]),
                        'disp':data_mom2[y,x], 'e_disp':0.0}
                    index_dict_stacked+=1
                    
                count+=1
                                    
                # ############## PLOTTING    
                # indices_wosignal = np.argwhere((self.spec_axis<data_vf_secondary[y,x]-3*data_mom2[y,x]) | (self.spec_axis>data_vf_secondary[y,x]+3*data_mom2[y,x]))
                # rms  = np.nanstd(data_cube[:,y,x][indices_wosignal])
                # peak = np.nanmax(data_cube[:,y,x])
                
                # if peak/rms<5: continue
                # count_plot+=5               
                # # if count_plot%100: continue                
                
                
                # print(peak, rms, peak/rms)
                    
                # import pylab as plt
                # import matplotlib
                # matplotlib.use('TkAgg')
                
                # fig, axs = plt.subplots(nrows=3)
                # plt.rcParams['hatch.linewidth']=4
                # ax=axs[0]
                # ax.axhline(0,color='gray',alpha=0.5)
                # print(data_cube[:,y,x])
                
                # ax.step(self.spec_axis, data_cube[:,y,x],color='gray',where='mid')
                # ax.fill_between(self.spec_axis, data_cube[:,y,x], step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                # ax.axvline(data_vf_secondary[y,x])
                # ax.axvspan(data_vf_secondary[y,x]-3*data_mom2[y,x],data_vf_secondary[y,x]+3*data_mom2[y,x], color='gray',alpha=0.1)

                # fig.suptitle("({}, {})".format(x, y))
                # ax.set_xlabel(r"Velocity [km $s^{-1}]$", color='white', fontsize=20, labelpad=-3)
                # ax.set_ylabel(r"Intensity [mJy/beam]",   color='white', fontsize=20)
                
                # xs = np.linspace(self.spec_axis.min(),self.spec_axis.max(),100)
                
                # ax=axs[1]
                # ax.axhline(0,color='gray',alpha=0.5)
                # ax.step(        xx*np.abs(self.chansep), shifted, where='mid', color='gray')
                # ax.fill_between(xx*np.abs(self.chansep), shifted, step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                # ax.set_ylim(axs[0].get_ylim())
                
                # ax=axs[2]
                # ax.axhline(0,color='gray',alpha=0.5)
                # ax.step(        xx*np.abs(self.chansep), yy, where='mid', color='gray')
                # ax.fill_between(xx*np.abs(self.chansep), yy, step='mid', hatch=r"//", color='lightgrey', edgecolor='white')
                
                # plt.show()
                # plt.close(fig)
        
                # import pylab as plt
                prog_perc = count/len_data*100
                if prog_perc // 10 > last_reported:
                    last_reported = prog_perc // 10
                    self.writestat(f"{self.stat} {prog_perc:.0f}% .")
        
        # import pylab as plt
        # fig,axs=plt.subplots(ncols=3, sharex=True, sharey=True)
        # ax = axs[0]
        # ax.imshow(self.data_mask, interpolation='none')
        # ax = axs[1]
        # ax.imshow(data_mask, interpolation='none')
        # ax = axs[2]
        # ax.imshow(map_stacked, interpolation='none')
        # # plt.savefig('/home/mskim/workspace/research/data/test/tmp.png', dpi=200)
        # ax.invert_yaxis()
        # plt.show()
        # raise
        
        yy  =  yy * (u.Jy/u.beam)
        e_y = e_y * (u.Jy/u.beam)
        
        yy  =          yy.to(u.Jy/u.sr, equivalencies=u.beam_angular_area(self.beam)) * self.area_pixl
        e_y = std_channel.to(u.Jy/u.sr, equivalencies=u.beam_angular_area(self.beam)) * self.area_pixl * np.sqrt(NN / self.pix_per_beam)
        e_y[np.argwhere(e_y==0)] = np.median(e_y)

        xx,yy,e_y = xx*np.abs(self.chansep), yy, e_y
        
        df = pd.DataFrame()
        df['x']   = xx#.value
        df['y']   = yy.value
        df['e_y'] = e_y.value
        df['N']   = NN
        
        # df['y'] / df['y']#/df['N']
        
        pixels_in_beam = self.beam.sr / (self.cd*self.cd).to(u.sr)
        # print(pixels_in_beam)
        df = df.loc[df['N']>pixels_in_beam].reset_index(drop=True)
        
        # df = df.loc[df['N']>np.max(df['N'])*0.50].reset_index(drop=True)
        # df = df.loc[df['N']==np.max(df['N'])].reset_index(drop=True)

        self.xx  = df['x']
        self.yy  = df['y']
        self.e_y = df['e_y']
    
        self.df_stacked = df
        # self.list_disps = self.list_disps
        
    def save_df_stacked(self, path_to_save):
        self.df_stacked.to_string(path_to_save, index=False)
        
    # def save_list_disps(self, path_to_save):
    #     np.save(path_to_save, self.list_disps)
    
    

    
if __name__=='__main__':
    

    
    multipliers = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    
    # for survey in ['LITTLE_THINGS','THINGS','VLA-ANGST','AVID','VIVA']:
    for survey in ['VIVA']:
    
        paths_cube = natsorted(list(Path(f'/home/mandu/workspace/research/data/{survey}_halfbeam').glob('*/cube.fits')))
        
        # for i, multiplier in enumerate(multipliers):
        #     suffix = f'_I{multiplier}r25'
        #     run_makemask_ellipse(paths_cube, multiplier_radius=multiplier, path_df='/home/mandu/workspace/research/data/catalog/cat_diameters.csv', suffix=suffix)
        
        pbar = tqdm(paths_cube)
        
        for path_cube in pbar:
            
            name_cube = path_cube.parent.name
            path_clfy = path_cube.parent/'segmts_merged_n_classified.3'
            
            pbar.set_description(f'{survey},{name_cube}')
            
            df = pd.DataFrame()
            
            for i, multiplier in enumerate(multipliers):
                
                suffix = f'_I{multiplier}r25'
                
                path_mask = path_cube.parent/f'mask{suffix}.fits'
                
                sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask)
                if name_cube=='NGC1569':
                    sns.mask_specrange(-20*(u.km/u.s), 20*(u.km/u.s))      
                sns.pbar = True
                sns.run()
                
                df_stacked = sns.df_stacked
                df_stacked[suffix[1:]] = df_stacked['y']
                
                if i==0: df = df_stacked[['x',suffix[1:]]]
                else:
                    df = pd.merge(df, df_stacked[['x',suffix[1:]]])
            
            # print(df.describe())
            path_df_out = path_cube.parent/'df_stacked.csv'
            df.to_string(path_df_out, index=False)
                    
                
                
else:
    from .subroutines_shiftnstack import argfind_nearest, shifter