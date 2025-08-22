import glob
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from radio_beam.beam import Beam
from routines_baygaudpi import gauss
from spectral_cube import SpectralCube
from spectral_cube.io.core import StokesWarning
from tqdm import tqdm



warnings.simplefilter("ignore", StokesWarning)



class ShiftnStack_hermite:
    
    def __init__(self, path_cube, path_hermite, path_mask=None, path_vf_secondary=None):
        path_cube = Path(path_cube)
        path_hermite = Path(path_hermite)
        if(path_mask is not None): path_mask = Path(path_mask)
        
        self.path_cube = path_cube
        self.path_mask = path_mask
        
        self.name_cube = path_cube.parent.name
        
        hedr_cube = fits.getheader(path_cube)
        data_cube = fits.getdata(path_cube) #* (u.Jy/u.beam)
        if(len(data_cube.shape)>3): data_cube = data_cube[0,:,:,:]
        
        self.hedr_cube = hedr_cube
        self.data_cube = data_cube 
        
        output_hermite = np.load(path_hermite)
        self.data_vf   = (output_hermite[5,:,:] * u.m/u.s).to(u.km/u.s).value
        self.dict_disp = {}
        data_disp = (output_hermite[2,:,:]*u.m/u.s).to(u.km/u.s).value

        if path_vf_secondary is not None:
            path_vf_secondary = Path(path_vf_secondary)
            data_vf_secondary = (fits.getdata(path_vf_secondary)*(u.m/u.s)).to(u.km/u.s).value
            if len(data_vf_secondary.shape)>2: data_vf_secondary = data_vf_secondary[0,:,:]
            data_vf_secondary = data_vf_secondary
        else:
            data_vf_secondary = np.full((hedr_cube['NAXIS2'],hedr_cube['NAXIS1']),np.nan)
        self.data_vf_secondary = data_vf_secondary
        
        if(path_mask is not None):
            path_mask = Path(path_mask)
            data_mask = fits.getdata(path_mask)
            if(len(data_mask.shape)>2):
                data_mask = data_mask[0,:,:]
            self.data_vf = np.where(np.isfinite(data_mask), self.data_vf, np.nan)
            # data_cube = np.where(np.isfinite(np.broadcast_to(data_mask, data_cube.shape)), data_cube, np.nan)
        else:
            data_mask = np.ones((hedr_cube['NAXIS2'],hedr_cube['NAXIS1']))
        self.data_mask = data_mask
            
        data_disp = data_disp[np.isfinite(data_disp)]
        for i in range(len(data_disp.flatten())):
            self.dict_disp[i] = {'disp':data_disp.flatten()[i]}
        
        testchannel = 0
        std_channel = 0
        while(std_channel==0):#*u.Jy/u.beam):
            std_channel = np.nanstd(data_cube[testchannel,:,:])
            testchannel+=1
        self.std_channel = std_channel * u.Jy/u.beam
                        
        spec_axis = (SpectralCube.read(path_cube)).with_spectral_unit(u.km/u.s, velocity_convention='optical').spectral_axis.value
        chansep = np.diff(spec_axis)[0]
        
        self.spec_axis = spec_axis
        self.chansep   = chansep
        
        beam = Beam.from_fits_header(hedr_cube)
        self.beam = beam
        self.bM = beam.major.to(u.arcsec)
        self.bm = beam.minor.to(u.arcsec)
        self.cd = (hedr_cube['CDELT2']*u.degree).to(u.arcsec)
        self.len_sa = len(self.spec_axis)
        
        self.pbar = False
        
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
       
        data_cube = np.float64(self.data_cube.copy())
        dict_data = {}
        data_vf = self.data_vf
        for y,x in np.argwhere(np.isfinite(self.data_vf)):
            coord = '{},{}'.format(x,y)
            dict_data[coord] = {
                'velo':data_vf[y,x]
            }
            
        data_mask = self.data_mask
        data_vf_secondary = self.data_vf_secondary
        
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
            
        data_cube[np.isnan(data_cube)] = np.random.normal(0,std_channel.value)#*data_cube.unit
    
        if self.pbar: pbar = tqdm(total=len(dict_data))
        for coord in dict_data.keys():
            x,y = map(int, coord.split(','))
            
            data_mask[y,x] = np.nan
            
            dict_cord = dict_data[coord]
                
            shift = dict_cord['velo']/self.chansep
            
            index = argfind_nearest(sa_div_chan, shift)
            
            mod = index_centr - index
            shifted = shifter(np.float64(data_cube[:,y,x]), mod, lenx)
            yy  += shifted
            NN  += (shifted != 0).astype(np.uint8)
        
            if self.pbar: pbar.update(1)
        
        if stack_secondary:
            for y,x in np.argwhere((np.isfinite(data_mask)) & (np.isfinite(data_vf_secondary))):
                shift = data_vf_secondary[y,x]/self.chansep
                index = argfind_nearest(sa_div_chan, shift)
                mod = index_centr - index
                shifted = shifter(data_cube[:,y,x], mod, lenx)
                yy  += shifted
                NN  += (shifted != 0).astype(np.uint8)
        
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
        
        pixels_in_beam = self.beam.sr / (self.cd*self.cd).to(u.sr)
        df = df.loc[df['N']>pixels_in_beam].reset_index(drop=True)
        
        # df = df.loc[df['N']>0].reset_index(drop=True)

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
    import pandas as pd
    from natsort import natsorted
    from run_makemask_ellipse import run_makemask as run_makemask_ellipse
    from tqdm import tqdm
    
    multipliers = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    
    # for survey in ['LITTLE_THINGS','THINGS','VLA-ANGST','AVID','VIVA']:
    for survey in ['VIVA']:
    
        paths_cube = natsorted(list(Path(f'/home/mandu/workspace/research/data/{survey}_halfbeam').glob('VCC1581/cube.fits')))
        
        pbar = tqdm(paths_cube)
        
        for path_cube in pbar:
            
            name_cube = path_cube.parent.name
            path_clfy = path_cube.parent/'hermite.npy'
            path_df_out = path_cube.parent/'df_stacked.csv'
            
            pbar.set_description(f'{survey},{name_cube}')
            
            # df = pd.DataFrame()
            df = pd.read_csv(path_df_out, sep='\s+')
            
            for i, multiplier in enumerate(multipliers):
                
                suffix = f'_I{multiplier}r25'
                path_mask = path_cube.parent/f'mask{suffix}.fits'
                
                suffix = f'_I{multiplier}r25_her'
                
                sns = ShiftnStack_hermite(path_cube, path_clfy, path_mask=path_mask)
                if name_cube=='NGC1569':
                    sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))      
                sns.pbar = True
                sns.run()
                
                df_stacked = sns.df_stacked
                df_stacked[suffix[1:]] = df_stacked['y']
                
                # df = pd.merge(df, df_stacked[['x',suffix[1:]]], on='x', how='left')
                
                df[suffix[1:]] = df_stacked['y']
            
            
            # print(df.describe())
        
            
            df.to_string(path_df_out, index=False)

else:
    from .subroutines_shiftnstack import argfind_nearest, shifter