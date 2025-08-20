from pathlib import Path
from tqdm import tqdm
from tool_makemask_ellipse import makemask
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.io import fits
import warnings
from astropy.wcs.wcs import FITSFixedWarning
from radio_beam import Beam

# Suppress the FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

def run_makemask(paths_cube, multiplier_radius_center, width, col_radius,
                 path_df='/home/mandu/workspace/research/data/catalog/cat_diameters.csv'):

    df_diam = pd.read_csv(path_df, sep=r'\s+')
    df_diam['RHI_arcsec'] = (df_diam['RHI(kpc)'] / 1000) / df_diam['D'] * (180 / np.pi) * 3600

    for path_cube in paths_cube:
        path_cube = Path(path_cube)
        galname = path_cube.parent.name
        if galname == 'VCC1091+98':
            galname = 'VCC1091'

        header = fits.getheader(path_cube)
        cd2 = (header['CDELT2'] * u.deg).to(u.arcsec)

        try:
            loc = np.argwhere(df_diam['Name'] == galname).item()
        except ValueError:
            raise ValueError(f'Galaxy name not found in catalog: {galname}')

        # Galaxy ellipse parameters
        ra = df_diam.loc[loc, 'RA']
        dec = df_diam.loc[loc, 'Dec']
        axr = df_diam.loc[loc, 'b/a']
        pa = df_diam.loc[loc, 'PA'] * u.deg

        # Beam width calculation
        if width == 'beam' or width == 'halfbeam' or width=='2*beam':
            beam = Beam.from_fits_header(header)
            bmean = (beam.major + beam.minor) / 2.0
            pm = bmean / 2.0
            if width == 'halfbeam':
                pm = pm / 2.0
            if width =='2*beam':
                pm = pm * 2.0
        else:
            pm = None  # Will be set later

        # Radius calculation
        if col_radius == 'r25':
            radius = df_diam.loc[loc, 'r25'] * u.arcsec
            radtag = 'r25'
        elif col_radius == 'RHI(kpc)':
            radius = df_diam.loc[loc, 'RHI_arcsec'] * u.arcsec
            radtag = 'RHI'
        else:
            raise ValueError(f'Unknown col_radius value: {col_radius}')

        # Handle missing radius
        if not np.isfinite(radius):# or not np.isfinite(df_diam.loc[loc, 'r25']):
            radius_inner = radius_outer = np.nan
        else:
            if pm is None:
                pm = width * radius / 2.0
                if pm < cd2:
                    pm = cd2

            radius_inner = radius * multiplier_radius_center - pm
            radius_outer = radius * multiplier_radius_center + pm

        # Create mask
        if radius_inner < 0 * u.arcsec or not np.isfinite(radius_inner):
            data_mask = makemask(path_cube, ra, dec, radius_outer, axr, pa, maskwhere='I')
        else:
            data_mask_inner = makemask(path_cube, ra, dec, radius_inner, axr, pa, maskwhere='O')
            data_mask_outer = makemask(path_cube, ra, dec, radius_outer, axr, pa, maskwhere='I')
            data_mask = np.where(np.isfinite(data_mask_inner) & np.isfinite(data_mask_outer), 1, np.nan)

        # Save mask
        maskdir = path_cube.parent / 'mask'
        maskdir.mkdir(exist_ok=True)
        writename = maskdir / f'mask_R{multiplier_radius_center:.2f}{radtag}.fits'
        fits.writeto(writename, data_mask, overwrite=True)


if __name__ == '__main__':
    run_makemask(
        ['/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/LGS3/cube.fits'],
        multiplier_radius_center=1.0,
        width='halfbeam',  # <-- Now supported
        col_radius='RHI(kpc)'
    )
