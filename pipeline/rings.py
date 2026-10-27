"""Beam-spaced optical annuli and auditable stopping decisions."""
import math

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .models import StackJob
from .optical import read_optical_geometry


def elliptical_radius(wcs, geometry, x, y):
    sky = wcs.pixel_to_world(x, y).icrs
    sky = SkyCoord(ra=sky.ra, dec=sky.dec, frame='icrs')
    center = SkyCoord(geometry['Diam_ra'] * u.deg, geometry['Diam_dec'] * u.deg)
    east, north = center.spherical_offsets_to(sky)
    pa = np.deg2rad(geometry['Diam_pa'])
    major = east.arcsec * np.sin(pa) + north.arcsec * np.cos(pa)
    minor = east.arcsec * np.cos(pa) - north.arcsec * np.sin(pa)
    return np.hypot(major, minor / geometry['Diam_ba'])


def usable_pixels(galaxy):
    # Use the stacker's actual eligibility rules, including HERMOM fallback and
    # the channel-width dispersion cut, rather than a different HI proxy.
    from shiftnstack import ShiftnStack
    from .galaxy_settings import dict_config_stack
    stack = ShiftnStack(
        galaxy.cube, None, path_vf_secondary=galaxy.secondary_vf,
        secondary_mode=galaxy.secondary_mode,
        correct_vf_secondary=dict_config_stack.get(galaxy.name, {}).get('correct_secondary_vf', False),
    )
    return stack.secondary_pixel_mask()


def beam_ring_jobs(config, galaxy):
    settings = config.masks.rings
    stop = settings.stop
    header = fits.getheader(galaxy.cube)
    major, minor = float(header.get('BMAJ', np.nan)), float(header.get('BMIN', np.nan))
    if not np.isfinite([major, minor]).all() or min(major, minor) <= 0:
        raise ValueError(f'{galaxy.name}: beam rings require positive FITS BMAJ and BMIN')
    beam = {'mean': math.sqrt(major * minor), 'major': max(major, minor),
            'minor': min(major, minor)}[settings.beam_fwhm] * 3600
    geometry = read_optical_geometry(config.masks.catalog, galaxy.name)
    ropt = float(geometry['Diam']) / 2
    wcs = WCS(header).celestial
    y, x = np.indices((header['NAXIS2'], header['NAXIS1']))
    rho = elliptical_radius(wcs, geometry, x, y)
    width = settings.width_beams * beam
    start = settings.start_beams * beam
    usable = usable_pixels(galaxy) if stop.mode == 'filling_factor' else None
    if usable is not None and usable.shape != rho.shape:
        raise ValueError(f'{galaxy.name}: usable mask and cube spatial shape differ')
    limit = stop.multiplier * (ropt if stop.radius == 'Ropt' else beam)
    if usable is not None:
        # Past the last on-image radius all rings are empty; this finite bound
        # also allows consecutive_below to be satisfied near the image edge.
        finite = rho[np.isfinite(rho)]
        if not finite.size:
            raise ValueError(f'{galaxy.name}: no finite sky coordinates')
        limit = max(start, float(finite.max())) + (stop.consecutive_below + 1) * width
    if math.ceil(max(0, limit - start) / width) > 10000:
        raise ValueError(f'{galaxy.name}: more than 10000 rings; increase width_beams')
    jobs, records = [], []
    below = []
    index = 0
    inner = start
    while inner < limit and not np.isclose(inner, limit, rtol=1e-12, atol=1e-10):
        outer = start + (index + 1) * width
        if stop.mode == 'radius' and outer > limit and not np.isclose(outer, limit, rtol=1e-12, atol=1e-10):
            if settings.final_ring == 'drop':
                records.append(dict(Name=galaxy.name, suffix='-', inner_arcsec=inner,
                                    outer_arcsec=limit, inner_beams=inner / beam,
                                    outer_beams=limit / beam, inner_Ropt=inner / ropt,
                                    outer_Ropt=limit / ropt, beam_arcsec=beam,
                                    retained=False, reason='partial_dropped'))
                break
            outer = limit
        elif stop.mode == 'radius' and np.isclose(outer, limit, rtol=1e-12, atol=1e-10):
            outer = limit
        selection = np.isfinite(rho) & (rho >= inner) & (rho < outer)
        selected = int(selection.sum())
        total, filled, fraction = selected, None, None
        if usable is not None:
            # Count the entire annulus on a virtual extension of the image grid.
            # Outside-image pixels therefore count as unfilled, not as missing
            # from the denominator. Work in tiles to bound temporary memory.
            center = SkyCoord(geometry['Diam_ra'] * u.deg, geometry['Diam_dec'] * u.deg)
            x0, y0 = wcs.world_to_pixel(center)
            scale = min(np.linalg.svd(wcs.pixel_scale_matrix, compute_uv=False)) * 3600
            extent = int(math.ceil(outer / scale * 1.2)) + 3
            # Verify the local-scale estimate against the actual sky transform,
            # expanding for projections/distortion until the ellipse is enclosed.
            for _ in range(12):
                left, right = math.floor(x0) - extent, math.ceil(x0) + extent
                bottom, top = math.floor(y0) - extent, math.ceil(y0) + extent
                xs, ys = np.arange(left, right + 1), np.arange(bottom, top + 1)
                edge_x = np.concatenate([xs, xs, np.full_like(ys, left), np.full_like(ys, right)])
                edge_y = np.concatenate([np.full_like(xs, bottom), np.full_like(xs, top), ys, ys])
                edge_r = elliptical_radius(wcs, geometry, edge_x, edge_y)
                if np.isfinite(edge_r).all() and np.min(edge_r) > outer:
                    break
                extent *= 2
            else:
                raise ValueError(f'{galaxy.name}: cannot enclose annulus in finite sky coordinates')
            total = 0
            for ystart in range(math.floor(y0) - extent, math.ceil(y0) + extent + 1, 256):
                for xstart in range(math.floor(x0) - extent, math.ceil(x0) + extent + 1, 256):
                    yy, xx = np.mgrid[ystart:min(ystart + 256, math.ceil(y0) + extent + 1),
                                      xstart:min(xstart + 256, math.ceil(x0) + extent + 1)]
                    radius = elliptical_radius(wcs, geometry, xx, yy)
                    total += int(((radius >= inner) & (radius < outer)).sum())
            total = max(total, selected)
            filled = int((selection & usable).sum())
            fraction = filled / total if total else 0.0
        suffix = f'_R{inner / beam:.12g}-{outer / beam:.12g}beam'
        record = dict(Name=galaxy.name, suffix=suffix, inner_arcsec=inner,
                      outer_arcsec=outer, inner_beams=inner / beam, outer_beams=outer / beam,
                      inner_Ropt=inner / ropt, outer_Ropt=outer / ropt, beam_arcsec=beam,
                      selected_pixels=selected, total_pixels=total, usable_pixels=filled,
                      filling_factor=fraction, retained=selected > 0 and filled != 0,
                      reason='retained' if selected and filled != 0 else 'empty')
        records.append(record)
        if record['retained']:
            jobs.append(StackJob(galaxy, 'ring', suffix,
                                 galaxy.cube.parent / f'NGFIT/mask/mask{suffix}.fits',
                                 inner_arcsec=inner, outer_arcsec=outer, beam_arcsec=beam))
        if fraction is not None:
            below = below + [record] if fraction == 0 or fraction < stop.min_fraction else []
            if len(below) >= stop.consecutive_below:
                rejected = below[:-1] if stop.include_trigger_ring else below
                for row in rejected:
                    row.update(retained=False, reason='filling_factor_stop')
                if stop.include_trigger_ring and record['retained']:
                    record['reason'] = 'trigger_included'
                rejected_keys = {row['suffix'] for row in rejected}
                jobs = [job for job in jobs if job.suffix not in rejected_keys]
                break
        index += 1
        inner = start + index * width
    return jobs, records


def create_ring_mask(job, config, cache=None):
    key = ('ring_radius', job.galaxy.cube, config.catalog)
    if cache is not None and key in cache:
        radius, header = cache[key]
    else:
        geometry = read_optical_geometry(config.catalog, job.galaxy.name)
        cube_header = fits.getheader(job.galaxy.cube)
        wcs = WCS(cube_header).celestial
        y, x = np.indices((cube_header['NAXIS2'], cube_header['NAXIS1']))
        radius = elliptical_radius(wcs, geometry, x, y)
        header = wcs.to_header()
        if cache is not None:
            cache[key] = (radius, header)
    header = header.copy()
    header['RIN'] = (job.inner_arcsec, 'Inner semi-major radius, arcsec (inclusive)')
    header['ROUT'] = (job.outer_arcsec, 'Outer semi-major radius, arcsec (exclusive)')
    header['BEAMFWHM'] = (job.beam_arcsec, 'Reference beam FWHM, arcsec')
    header.add_history(f'Optical beam rings from {config.catalog}')
    selection = np.isfinite(radius) & (radius >= job.inner_arcsec) & (radius < job.outer_arcsec)
    job.mask.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(job.mask, np.where(selection, 1, np.nan).astype('float32'), header, overwrite=True)
