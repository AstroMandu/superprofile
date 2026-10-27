"""Complementary sky ellipses from the NEDLVS major-axis diameter (arcsec).

Masks select spatial pixels; ShiftnStack additionally applies its HI detection
and velocity-field validity masks. Outer means all detected HI outside Ropt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from .catalogs import resolve_catalog_name


class AmbiguousOpticalGeometryError(ValueError):
    """More than one catalog row has usable geometry for this selection."""


def read_optical_geometry(catalog, galaxy, *, center_only=False):
    """Select exactly one row valid for the geometry this mask requires."""
    table = pd.read_csv(catalog, sep=r'\s+')
    catalog_name = resolve_catalog_name(galaxy, set(table['objname']))
    rows = table.loc[table['objname'] == catalog_name]
    columns = ['Diam_ra', 'Diam_dec']
    if not center_only:
        columns += ['Diam', 'Diam_ba', 'Diam_pa']
    values = rows.reindex(columns=columns).apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(values).all(axis=1) & values['Diam_dec'].between(-90, 90)
    if not center_only:
        valid &= (values['Diam'] > 0) & (values['Diam_ba'] > 0) & (values['Diam_ba'] <= 1)
    usable = values.loc[valid]
    if len(usable) > 1:
        raise AmbiguousOpticalGeometryError(
            f'{galaxy}: expected one valid optical catalog entry, found {len(usable)} '
            f'(from {len(rows)} matching rows)')
    if usable.empty:
        kind = 'center' if center_only else 'ellipse'
        raise ValueError(f'{galaxy}: missing or invalid optical {kind}; '
                         f'no valid entry among {len(rows)} matching rows')
    return usable.iloc[0].astype(float)


def create_optical_masks(cube, catalog, multiplier=1.0, cache=None, combined=False):
    cube = Path(cube)
    key = ('ellipse', cube, Path(catalog), float(multiplier), combined)
    if cache is not None and key in cache:
        return cache[key]
    geometry = read_optical_geometry(catalog, cube.parent.name)
    header = fits.getheader(cube)
    wcs = WCS(header).celestial
    y, x = np.indices((header['NAXIS2'], header['NAXIS1']))
    sky = wcs.pixel_to_world(x, y).icrs
    # Drop inherited FK5 equinox attributes after the ICRS transformation.
    sky = SkyCoord(ra=sky.ra, dec=sky.dec, frame="icrs")
    center = SkyCoord(geometry['Diam_ra'] * u.deg, geometry['Diam_dec'] * u.deg)
    east, north = center.spherical_offsets_to(sky)
    east, north = east.arcsec, north.arcsec
    pa = np.deg2rad(geometry['Diam_pa'])
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = east * np.cos(pa) - north * np.sin(pa)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Optical radius multiplier must be finite and positive')
    radius = geometry['Diam'] / 2 * multiplier
    rho2 = (major / radius)**2 + (minor / (radius * geometry['Diam_ba']))**2
    valid = np.isfinite(rho2)
    selections = {'I': valid & (rho2 <= 1), 'O': valid & (rho2 > 1)}
    out_header = wcs.to_header()
    out_header['ROPT'] = (radius, 'Optical semi-major radius, arcsec')
    out_header['AXRATIO'] = float(geometry['Diam_ba'])
    out_header['OPT_PA'] = (float(geometry['Diam_pa']), 'Position angle east of north, deg')
    out_header['OPT_RA'] = float(geometry['Diam_ra'])
    out_header['OPT_DEC'] = float(geometry['Diam_dec'])
    out_header.add_history(f'Optical ellipse from {Path(catalog).resolve()}')
    paths = {}
    maskdir = cube.parent / 'NGFIT' / 'mask'
    maskdir.mkdir(parents=True, exist_ok=True)
    if combined:
        path = maskdir / f'mask_IO{float(multiplier)}Ropt.fits'
        values = np.where(valid, (rho2 <= 1).astype('float32'), np.nan).astype('float32')
        out_header['REGION'] = 'IO'
        out_header['MASKONE'] = 'inner'
        out_header['MASKZERO'] = 'outer'
        fits.writeto(path, values, out_header, overwrite=True)
        paths['IO'] = path
        if cache is not None:
            cache[('selection', path)] = (values, out_header)
            cache[key] = paths
        return paths
    for region, selection in selections.items():
        path = maskdir / f'mask_{region}{float(multiplier)}Ropt.fits'
        region_header = out_header.copy()
        region_header['REGION'] = 'inner' if region == 'I' else 'outer'
        fits.writeto(path, np.where(selection, 1, np.nan).astype('float32'), region_header, overwrite=True)
        paths[region] = path
        if cache is not None:
            cache[('selection', path)] = (selection, region_header)
    if cache is not None:
        cache[key] = paths
    return paths


def create_optical_angle_mask(job, catalog, cache=None):
    """Create a center-based wedge, optionally intersected with a NEDLVS ellipse.

Angular convention matches the legacy helper: 0 degrees is image +y, and
positive angles rotate toward image -x. These are not deprojected disk angles.
"""
    key = ('angle_grid', job.galaxy.cube, Path(catalog))
    if cache is not None and key in cache:
        geometry, base_header, phi = cache[key]
    else:
        geometry = read_optical_geometry(catalog, job.galaxy.name, center_only=True)
        header = fits.getheader(job.galaxy.cube)
        wcs = WCS(header).celestial
        center = SkyCoord(geometry['Diam_ra'] * u.deg, geometry['Diam_dec'] * u.deg)
        x0, y0 = wcs.world_to_pixel(center)
        y, x = np.indices((header['NAXIS2'], header['NAXIS1']))
        phi = np.rad2deg(np.arctan2(y - y0, x - x0))
        base_header = wcs.to_header()
        if cache is not None:
            cache[key] = (geometry, base_header, phi)
    delta = (phi - 90 - job.angle + 180) % 360 - 180
    selection = np.abs(delta) <= job.angle_width / 2
    output_header = base_header.copy()
    values = None
    if job.radial_region:
        masks = create_optical_masks(job.galaxy.cube, catalog, multiplier=job.multiplier,
                                     cache=cache, combined=True)
        if cache is not None:
            values, radial_header = cache[('selection', masks['IO'])]
            output_header = radial_header.copy()
        else:
            values = fits.getdata(masks['IO'])
            output_header = fits.getheader(masks['IO'])
    else:
        output_header['OPT_RA'] = float(geometry['Diam_ra'])
        output_header['OPT_DEC'] = float(geometry['Diam_dec'])
        output_header.add_history(f'Angular center from {Path(catalog).resolve()}')
    output_header['ANGCENT'] = (job.angle, 'Image-plane angle from +y toward -x, deg')
    output_header['ANGWIDTH'] = (job.angle_width, 'Angular wedge width, deg')
    job.mask.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(job.mask, np.where(selection, values if values is not None else 1,
                                  np.nan).astype('float32'), output_header, overwrite=True)
