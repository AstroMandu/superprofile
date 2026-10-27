import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pylab as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from photutils.aperture import EllipticalAperture
from regions import EllipsePixelRegion, PixCoord

warnings.filterwarnings(
    "ignore",
    category=FITSFixedWarning,
)

def makemask(path_cube, angle_center, angle_width, path_df, path_mask=None, savename=None, colname_RA='RA', colname_DEC='Dec', preserve_mask_values=False):
    
    df = pd.read_csv(path_df, sep='\\s+')

    path_cube = Path(path_cube)
    wdir = path_cube.parent
    galname = wdir.name
    if galname=='VCC1091+98': galname='VCC1091'
    if 'VCC152' in galname: galname='VCC152'
    if 'VCC699' in galname: galname='VCC699'
    
    dfloc = df.loc[df['Name']==galname].reset_index(drop=True)
    
    hedr = fits.getheader(path_cube)
    data = fits.getdata(path_cube)
    if len(data.shape)>3: data=data[0,0,:,:]
    else: data=data[0,:,:]

    wcs = WCS(hedr)

    # # if path_cube.parent.parent.parent.name=='Rory':
    # if path_cube.parent.parent.parent.parent.name=='Rory':
    #     # path_mom0 = path_cube.parent / f'cube_mom0.fits'
    #     # data_mom0 = fits.getdata(path_mom0)
    #     # y0, x0 = np.unravel_index(np.argmax(data_mom0), data_mom0.shape)
    #     # print('Using peak pixel from mom0 as center: ', path_cube.parent.name, x0, y0)
        
    #     coords = SkyCoord('0h0m0s','0d0m0s',frame='icrs')
    #     x0, y0 = PixCoord.from_sky(coords, wcs).x, PixCoord.from_sky(coords, wcs).y
        
    #     if path_cube.parent.name=='left':  x0-=6
    #     if path_cube.parent.name=='right': x0+=6
    #     if path_cube.parent.name=='up':    y0+=6
    #     if path_cube.parent.name=='down':  y0-=6
    
    if False:
        pass

    else:
        col_dec = colname_DEC if colname_DEC in dfloc.columns else ('DEC' if 'DEC' in dfloc.columns else colname_DEC)
        # RA/Dec are either sexagesimal strings (e.g. '12h14m53.54s', '09:55:33.2')
        # or plain decimal-degree numbers, depending on the source catalog.
        if pd.api.types.is_numeric_dtype(dfloc[colname_RA]):
            coords = SkyCoord(dfloc[colname_RA],dfloc[col_dec],unit=(u.deg,u.deg),frame='icrs')
        else:
            coords = SkyCoord(dfloc[colname_RA],dfloc[col_dec],frame='icrs')
        x0, y0 = PixCoord.from_sky(coords, wcs).x, PixCoord.from_sky(coords, wcs).y

    # # Step 1: Full ellipse mask
    # reg = EllipsePixelRegion(PixCoord.from_sky(coords, wcs),
    #                          width=rmaj_pix, height=rmin_pix, angle=rpa)
    # mask_ellipse = reg.to_mask(mode='center').to_image(data.shape).astype(bool)
    mask_ellipse = np.full_like(data,1)
    if path_mask is not None:
        mask_ellipse = fits.getdata(path_mask)
        if not preserve_mask_values:
            mask_ellipse = np.where(np.isfinite(mask_ellipse),1,0)

    # Step 2: Angular wedge mask
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]

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
    if preserve_mask_values:
        mask_combined = np.where(mask_angle, mask_ellipse, np.nan).astype('float32')
    else:
        mask_combined = np.logical_and(mask_ellipse,mask_angle)
        mask_combined = np.where(mask_combined==1, 1, np.nan)
    
    if os.path.exists(wdir/'NGFIT'/'mask')==False:
        (wdir/'NGFIT'/'mask').mkdir(parents=True, exist_ok=True)
    
    if savename is None:
        savename = wdir/'NGFIT/mask/mask_A{:0>3}_W{:0>3}.fits'.format(angle_center, angle_width)
    
    header = None
    if preserve_mask_values:
        header = fits.getheader(path_mask)
        header['ANGCENT'] = angle_center
        header['ANGWIDTH'] = angle_width
    fits.writeto(savename, mask_combined, header=header, overwrite=True)
    
# path_cube = '/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO53/cube_mom1.fits'
# makemask(path_cube, 90,30)
