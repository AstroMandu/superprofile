import glob
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from radio_beam.beam import Beam
from spectral_cube import SpectralCube
import pandas as pd
from routines_baygaudpi import gauss, read_ngfits
# from .class_Filter_genuine import Filter
import warnings
from spectral_cube.io.core import StokesWarning
from tqdm import tqdm



warnings.simplefilter("ignore", StokesWarning)


class ShiftnStack:
    
    def __init__(self, path_cube, path_classified, path_mask=None, path_vf_secondary=None, bgsub=True):
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
        
        testchannel = 0
        std_channel = 0
        while(std_channel==0):#*u.Jy/u.beam):
            std_channel = np.nanstd(data_cube[testchannel,:,:])
            testchannel+=1
        self.std_channel = std_channel * u.Jy/u.beam
           
        nopt = len(glob.glob(str(path_classified/"ngfit/*G*_*.0.fits")))
        
        map_ngau = fits.getdata(glob.glob(str(path_classified/"sgfit/*.7.fits"))[0])
        # 
        
        self.len_nopt = int(np.nansum(map_ngau))
        if bgsub:
            map_bkgr = fits.getdata(glob.glob(str(path_classified/"ngfit/*_1.3.fits"))[0])#*(u.Jy/u.beam)
            self.data_cube-= np.where(np.isfinite(map_bkgr),map_bkgr,0)
        
        self.spec_axis = spec_axis
        self.chansep = chansep
        self.nopt = nopt
        
        beam = Beam.from_fits_header(hedr_cube)
        self.beam = beam
        self.bM = beam.major.to(u.arcsec)
        self.bm = beam.minor.to(u.arcsec)
        self.cd = (hedr_cube['CDELT2']*u.degree).to(u.arcsec)
        self.len_sa = len(self.spec_axis)
        
        self.dict_disp = {}
        
        self.pbar = False
        self.bgsub = bgsub
        
    def mask_velrange(self, vel_mask_low, vel_mask_high):
        """
        Mask the cube based on velocity range
        """
        # Create a boolean mask for the spectral axis
        mask = (self.spec_axis > vel_mask_low.to(u.km/u.s).value) & (self.spec_axis < vel_mask_high.to(u.km/u.s).value)

        
        # Reshape mask to broadcast over the cube (assumes spectral axis is axis 0)
        mask_reshaped = mask[:, np.newaxis, np.newaxis]
        
        # Apply the mask
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
                                toreads=['velo','disp','psnr','bkgr','nois','e_disp'], 
                                path_mask=self.path_mask,
                                wo_unit=True)
        
        if self.bgsub:
            for coord in dict_data.keys():
                for g in dict_data[coord].keys():
                    dict_data[coord][g]['bkgr'] = 0.
        
        data_mask = self.data_mask
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
        
        len_specaxis = len(self.spec_axis)
        if self.chansep<0:
            xx = np.arange(len_specaxis-1, -len_specaxis, -1)
        else:
            xx  = np.arange(-len_specaxis+1, len_specaxis, 1)
        lenx = len(xx)
            
        yy  = np.zeros_like(xx, dtype=np.float64)#*(u.Jy/u.beam)
        e_y = np.zeros_like(xx, dtype=np.float64)#*(u.Jy/u.beam)
        NN  = np.zeros_like(xx)
        
        sa_div_chan = self.spec_axis / self.chansep
        index_centr = np.argwhere(xx==0).item()
        
        std_channel = self.std_channel
            
        # data_cube[np.isnan(data_cube)] = np.random.normal(0,std_channel.value)#*data_cube.unit
    
        # from tqdm import tqdm
        # for coord in tqdm(dict_data.keys()):
        if self.pbar: pbar = tqdm(total=len(dict_data))
        count = 0
        for coord in dict_data.keys():
            x,y = map(int, coord.split(','))
            dict_cord = dict_data[coord]
            
            data_mask[y,x] = np.nan

            for g in dict_cord.keys():
                
                dict_pg = dict_cord[g]
                if 'velo' not in dict_pg: continue
                # if self.filter_genuine(dict_pg)==False: continue
                
                data_subed = np.float64(data_cube[:,y,x].copy())
                # showplot=False
                # if len(dict_cord)>1: showplot=True
                for gother in dict_cord.keys():
                    if(g==gother): continue
                    dict_pgother = dict_cord[gother]
                    if 'velo' not in dict_pgother: continue
                    
                    model = gauss(self.spec_axis, dict_pgother['ampl'], dict_pgother['velo'], dict_pgother['disp'])+dict_pgother['bkgr']
                    data_subed -= model
                    data_subed += dict_pgother['bkgr']
                data_subed -= dict_pg['bkgr']
                                    
                shift = dict_pg['velo']/self.chansep
                
                index = argfind_nearest(sa_div_chan, shift)

                mod = index_centr - index
                shifted = shifter(data_subed, mod, lenx)
                yy  += shifted
                NN  += (shifted != 0).astype(np.uint8)
                                
                self.dict_disp[count] = {'disp':dict_pg['disp'], 'e_disp':dict_pg['e_disp']}
                count+=1
                
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
                    
                #     plt.show()
                #     plt.close(fig)
                    
            if self.pbar: pbar.update(1)
        
        if stack_secondary:
            
            import os
            
            
            if os.path.exists(self.path_cube.parent/'cube_mom2.fits'):
                data_mom2 = fits.getdata(self.path_cube.parent/'cube_mom2.fits')/1000.
            else: data_mom2 = None
            
            data_vf_secondary = np.where(data_mom2<self.chansep/2.355*2, np.nan, data_vf_secondary)
            
            for y,x in np.argwhere((np.isfinite(data_mask)) & (np.isfinite(data_vf_secondary))):
                shift = data_vf_secondary[y,x]/self.chansep
                index = argfind_nearest(sa_div_chan, shift)
                mod = index_centr - index
                shifted = shifter(data_cube[:,y,x], mod, lenx)
                yy  += shifted
                NN  += (shifted != 0).astype(np.uint8)
                
                if data_mom2 is not None:
                    self.dict_disp[count] = {'disp':data_mom2[y,x], 'e_disp':0.0}
                    count+=1
                    
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
                # # plt.close(fig)
                
        
        yy  = yy * (u.Jy/u.beam)
        e_y = e_y * (u.Jy/u.beam)
        yy  =   yy.to(u.Jy/u.arcsec**2, equivalencies=u.beam_angular_area(self.beam)) * (self.cd*self.cd)
        e_y = std_channel.to(u.Jy/u.arcsec**2, equivalencies=u.beam_angular_area(self.beam)) * (self.cd*self.cd) * np.sqrt(NN /((self.bM*self.bm*np.pi)/(self.cd*self.cd)))
        e_y[np.argwhere(e_y==0)] = np.median(e_y)

        xx,yy,e_y = xx*np.abs(self.chansep), yy, e_y
        
        df = pd.DataFrame()
        df['x']   = xx#.value
        df['y']   = yy.value
        df['e_y'] = e_y.value
        df['N']   = NN
        
        df = df.loc[df['N']>28.59].reset_index(drop=True)

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
    
    from subroutines_shiftnstack import argfind_nearest, shifter
    from run_makemask_ellipse import run_makemask as run_makemask_ellipse
    from tqdm import tqdm
    import pandas as pd
    from natsort import natsorted
    
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
                    sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))      
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