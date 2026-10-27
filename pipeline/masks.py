"""Generate only the masks required by pending stack jobs.

Legacy geometry routines remain available for older analysis scripts. This
module normalizes their radius-column conventions and centralizes filenames.
"""
from .optical import create_optical_masks, create_optical_angle_mask


def prepare_legacy_io_mask(cube, config, multiplier, cache=None):
    """Create one shared radial mask, reused by every wedge for this radius."""
    path = cube.parent / f'NGFIT/mask/mask_IO{float(multiplier)}{config.radius_column}.fits'
    key = ('legacy_io', path)
    if cache is None or key not in cache or not path.is_file():
        from run_makemask_ellipse import run_makemask
        run_makemask([cube], multiplier_radius=float(multiplier), col_radius=config.radius_column,
                     maskwheres=['IO'], path_df=config.catalog,
                     colname_RA=config.ra_column, colname_DEC=config.dec_column)
        if cache is not None:
            cache[key] = True
    return path


def prepare_mask(job, config, cache=None):
    if job.region == 'whole':
        return
    prepared_key = ('prepared_mask', job.mask)
    if job.mask_value is not None and cache is not None and prepared_key in cache and job.mask.is_file():
        return
    cube = job.galaxy.cube
    if job.region == 'ring' and job.inner_arcsec is not None:
        from .rings import create_ring_mask
        create_ring_mask(job, config, cache=cache)
    elif config.is_optical_catalog and job.region == 'angle':
        create_optical_angle_mask(job, config.catalog, cache=cache)
    elif config.is_optical_catalog:
        create_optical_masks(cube, config.catalog, multiplier=job.multiplier, cache=cache, combined=True)
    elif job.region in {'inner', 'outer'}:
        prepare_legacy_io_mask(cube, config, job.multiplier, cache=cache)
    elif job.region == 'ring':
        from run_makemask_ring import run_makemask
        column = 'RHI(kpc)' if config.radius_column == 'RHI' else config.radius_column
        run_makemask([cube], multiplier_radius_center=job.multiplier, width=config.ring_width,
                     col_radius=column, path_df=config.catalog,
                     colname_RA=config.ra_column, colname_DEC=config.dec_column)
    elif job.region == 'angle':
        from makemask_angle import makemask
        job.mask.parent.mkdir(parents=True, exist_ok=True)
        radial_mask = None
        if job.radial_region:
            radial_mask = prepare_legacy_io_mask(cube, config, job.multiplier, cache=cache)
        makemask(cube, angle_center=job.angle, angle_width=job.angle_width,
                 path_df=config.catalog, path_mask=radial_mask, savename=job.mask,
                 preserve_mask_values=bool(job.radial_region),
                 colname_RA=config.ra_column, colname_DEC=config.dec_column)
    if not job.mask.is_file():
        raise FileNotFoundError(f'Mask generator did not produce {job.mask}')
    if job.mask_value is not None and cache is not None:
        cache[prepared_key] = True


def write_geometry(config, galaxies, output):
    if not config.is_optical_catalog:
        return
    import pandas as pd
    from .optical import read_optical_geometry
    rows = []
    for galaxy in galaxies:
        geometry = read_optical_geometry(
            config.catalog, galaxy.name, center_only=not config.needs_ellipse).to_dict()
        rows.append({'Name': galaxy.name, 'Ropt_arcsec': geometry.get('Diam', float('nan')) / 2, **geometry})
    pd.DataFrame(rows).to_csv(output / 'optical_geometry.csv', index=False)
