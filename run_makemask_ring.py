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

def run_makemask(paths_cube, multiplier_radius_center, width, col_radius, path_df='/home/mandu/workspace/research/data/catalog/cat_diameters.csv'):

    df_diam = pd.read_csv(path_df, sep='\s+')
    df_diam['RHI_arcsec'] = (df_diam['RHI(kpc)']/1000) / df_diam['D'] * 180/np.pi * 3600
    
    for path_cube in paths_cube:
        path_cube = Path(path_cube)
        galname = path_cube.parent.name
        
        cd2 = (fits.getheader(path_cube)['CDELT2']*u.degree).to(u.arcsec)
        
        loc = np.argwhere(df_diam['Name']==galname).item()
        ra  = df_diam.loc[loc,'RA']
        dec = df_diam.loc[loc,'Dec']
        axr = df_diam.loc[loc,'b/a']
        pa  = df_diam.loc[loc,'PA']  * u.deg

        if col_radius=='r25':
            r25 = df_diam.loc[loc,'r25'] * u.arcsec
            if width*r25/2.<cd2: pm = cd2
            else: pm = width*r25/2.
            radius_inner = r25 * multiplier_radius_center - pm
            radius_outer = r25 * multiplier_radius_center + pm
            radtag = 'r25'
        if col_radius=='RHI(kpc)':
            RHI = df_diam.loc[loc,'RHI_arcsec'] * u.arcsec
            if width*RHI/2.<cd2: pm = cd2
            else: pm = width*RHI/2.
            radius_inner = RHI * multiplier_radius_center - pm
            radius_outer = RHI * multiplier_radius_center + pm
            radtag = 'RHI'
            
            if np.isfinite(df_diam.loc[loc,'r25'])==False: 
                radius_inner = np.nan
                radius_outer = np.nan
                
        if radius_inner<0*u.arcsec:
            data_mask = makemask(path_cube,ra,dec,radius_outer,axr,pa,maskwhere='I')
        else:
            data_mask_inner = makemask(path_cube,ra,dec,radius_inner,axr,pa,maskwhere='O')
            data_mask_outer = makemask(path_cube,ra,dec,radius_outer,axr,pa,maskwhere='I')
                
            data_mask = np.where(np.isfinite(data_mask_inner) & np.isfinite(data_mask_outer), 1, np.nan)
        
        maskdir = path_cube.parent/'mask'
        if os.path.exists(maskdir)==False:
            os.mkdir(maskdir)
        writename = path_cube.parent/'mask/mask_R{}{}.fits'.format(multiplier_radius_center,radtag)
        fits.writeto(writename, data_mask, overwrite=True)
        
        
if __name__=='__main__':
    run_makemask(['/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/LGS3/cube.fits'],
                 1.0,
                 0.1,
                 'RHI(kpc)')