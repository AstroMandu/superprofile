import os
from pathlib import Path

import numpy as np
import pandas as pd
import pylab as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import EllipticalAperture
from regions import EllipsePixelRegion, PixCoord

def makemask(path_cube, angle_center, angle_width, path_df, path_mask=None):
    
    df = pd.read_csv(path_df, sep='\s+')

    path_cube = Path(path_cube)
    wdir = path_cube.parent
    galname = wdir.name

    dfloc = df.loc[df['Name']==galname].reset_index(drop=True)

    coords = SkyCoord(dfloc['RA'],dfloc['Dec'],frame='icrs')

    hedr = fits.getheader(path_cube)
    data = fits.getdata(path_cube)
    if len(data.shape)>3: data=data[0,0,:,:]
    else: data=data[0,:,:]
    
    wcs = WCS(hedr)

    # # Step 1: Full ellipse mask
    # reg = EllipsePixelRegion(PixCoord.from_sky(coords, wcs),
    #                          width=rmaj_pix, height=rmin_pix, angle=rpa)
    # mask_ellipse = reg.to_mask(mode='center').to_image(data.shape).astype(bool)
    mask_ellipse = np.full_like(data,1)
    if path_mask is not None:
        mask_ellipse = fits.getdata(path_mask)
        mask_ellipse = np.where(np.isfinite(mask_ellipse),1,0)

    # Step 2: Angular wedge mask
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    x0, y0 = PixCoord.from_sky(coords, wcs).x, PixCoord.from_sky(coords, wcs).y

    # Shift coords to center
    dx = xx - x0
    dy = yy - y0
    phi = np.degrees(np.arctan2(dy, dx)) % 360  # angles in [0, 360)

    pa_min, pa_max = angle_center-angle_width/2., angle_center+angle_width/2.  # degrees, adjust to your use

    pa_min+=90
    pa_max+=90
    
    pa_min%=360
    pa_max%=360
    
    if pa_min < pa_max:
        mask_angle = (phi >= pa_min) & (phi <= pa_max)
    else:
        # wrap-around case (e.g., 330 to 30)
        mask_angle = (phi >= pa_min) | (phi <= pa_max)

    # Step 3: Combine masks
    mask_combined = np.logical_and(mask_ellipse,mask_angle)
    mask_combined = np.where(mask_combined==1, 1, np.nan)
    
    if os.path.exists(wdir/'mask')==False:
        os.mkdir(wdir/'mask')
    
    savename = wdir/'mask/mask_A{:0>3}_W{:0>3}.fits'.format(angle_center, angle_width)
    
    fits.writeto(savename, mask_combined, overwrite=True)
    
# path_cube = '/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO53/cube_mom1.fits'
# makemask(path_cube, 90,30)