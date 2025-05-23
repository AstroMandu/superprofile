from pathlib import Path
from tqdm import tqdm
from tool_makemask_ellipse import makemask
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.io import fits
import warnings
from astropy.wcs.wcs import FITSFixedWarning
import glob
import os

# Suppress the FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

def run_makemask(paths_cube, multiplier_radius, col_radius, maskwheres=['I','O'], path_df='/home/mandu/workspace/research/data/catalog/cat_diameters.csv'):

    df_diam = pd.read_csv(path_df, sep='\s+')
   
    for maskwhere in maskwheres:
        pbar = tqdm(paths_cube)
        for path_cube in pbar:
            
            path_cube = Path(path_cube)
            galname = path_cube.parent.name
            
            pbar.set_description('{} {} {}'.format(galname, maskwhere, multiplier_radius))
            
            if(path_cube.parent.parent.parent.name=='Rory'):
                print('[Makemask_ellipse] Using pre-defined config')
                ra = '23h59m26.1'
                dec = '0d08m41s'
                r25 = 155* u.arcsec * multiplier_radius
                axr = 0.5
                pa  = 90 * u.deg
            else:
            
                loc = np.argwhere(df_diam['Name']==galname).item()
                ra  = df_diam.loc[loc,'RA']
                dec = df_diam.loc[loc,'Dec']
                axr = df_diam.loc[loc,'b/a']
                pa  = df_diam.loc[loc,'PA']  * u.deg

                if col_radius=='r25':
                    radius = df_diam.loc[loc,'r25'] * u.arcsec * multiplier_radius
                    radtag = 'r25'
                if col_radius=='RHI(kpc)':
                    df_diam['RHI_arcsec'] = (df_diam['RHI(kpc)']/1000) / df_diam['D'] * 180/np.pi * 3600
                    radius = df_diam.loc[loc,'RHI_arcsec'] * u.arcsec * multiplier_radius
                    radtag = 'RHI'
                    
                    if np.isfinite(df_diam.loc[loc,'r25'])==False: radius = np.nan
            
            data_mask = makemask(path_cube, 
                                ra,dec,
                                radius,
                                axr,
                                pa,
                                maskwhere=maskwhere)
            
            maskdir = path_cube.parent/'mask'
            if os.path.exists(maskdir)==False:
                os.mkdir(maskdir)
            writename = path_cube.parent/'mask/mask_{}{}{}.fits'.format(maskwhere,multiplier_radius,radtag)
            fits.writeto(writename, data_mask, overwrite=True)