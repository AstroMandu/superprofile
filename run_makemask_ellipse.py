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

def run_makemask(paths_cube, multiplier_radius, col_radius, maskwheres=['I','O'], path_df='/home/mskim/workspace/research/data/catalog/cat_diameters.csv'):

    
   
    for maskwhere in maskwheres:
        for path_cube in paths_cube:
            
            path_cube = Path(path_cube)
            galname = path_cube.parent.name
            
            if(path_cube.parent.parent.parent.name=='Rory'):
                df_diam = pd.read_csv('/home/mskim/workspace/research/data/Rory/cat_diameters_rorysim.csv', sep='\\s+')
            else:
                df_diam = pd.read_csv(path_df, sep='\s+')
                
            try:
                loc = np.argwhere(df_diam['Name']==galname).item()
            except ValueError:
                raise ValueError('[Makemask_ellipse] {} not found in diameter catalog.'.format(galname))
            ra  = df_diam.loc[loc,'RA']
            dec = df_diam.loc[loc,'Dec']
            axr = df_diam.loc[loc,'b/a']
            pa  = df_diam.loc[loc,'PA']  * u.deg

            if col_radius=='r25':
                radius = df_diam.loc[loc,'r25'] * u.arcsec * multiplier_radius
                radtag = 'r25'
            if col_radius=='RHI':
                df_diam['RHI_arcsec'] = (df_diam['RHI(kpc)']/1000) / df_diam['D'] * 180/np.pi * 3600
                radius = df_diam.loc[loc,'RHI_arcsec'] * u.arcsec * multiplier_radius
                radtag = 'RHI'
            
                
                # if np.isfinite(df_diam.loc[loc,'r25'])==False: radius = np.nan
            
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