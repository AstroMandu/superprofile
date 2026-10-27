"""Validate inputs and construct deterministic jobs without creating files."""
from natsort import natsorted

from .config import PipelineConfig
from .catalogs import catalog_kind, resolve_catalog_name
from .models import GalaxyInput, StackJob

HER_MODES = {'HER', 'HERMOM'}
SECONDARY_ONLY_MODES = {'MOM', 'HER', 'HERMOM'}


def discover_galaxies(config: PipelineConfig):
    paths = ([config.data_dir / name / config.cube_name for name in config.galaxies]
             if config.galaxies is not None else list(config.data_dir.glob(f'*/{config.cube_name}')))
    if not paths:
        if config.galaxies is not None:
            print(f'[skip] No requested galaxies have {config.cube_name} inputs in {config.data_dir}', flush=True)
            return []
        raise FileNotFoundError(f'No {config.cube_name} inputs in {config.data_dir}')
    galaxies = []
    for cube in natsorted(paths):
        if not cube.is_file():
            print(f'[skip] {cube.parent.name}: missing cube {cube}', flush=True)
            continue
        folder = cube.parent
        mode = config.fit.mode
        secondary = folder / ('hermite.npy' if mode in HER_MODES else 'cube_mom1.fits')
        if config.fit.use_secondary_vf and not secondary.is_file():
            print(f'[skip] {folder.name}: missing {mode} velocity field {secondary}', flush=True)
            continue
        classified = None if mode in SECONDARY_ONLY_MODES else folder / f'segmts_merged_n_classified.{config.fit.key_classify}'
        if classified is not None and not classified.is_dir():
            if not config.fit.classify or not (folder / 'segmts').is_dir():
                print(f'[skip] {folder.name}: missing BAYGAUD input {classified}', flush=True)
                continue
        galaxies.append(GalaxyInput(folder.name, cube, classified, secondary if config.fit.use_secondary_vf else None,
                                    mode if mode in HER_MODES else 'MOM'))
    return galaxies


def plan_jobs(config, galaxies, ring_records=None):
    masks = config.masks
    jobs = []
    for galaxy in galaxies:
        for region in masks.regions:
            if region == 'whole':
                jobs.append(StackJob(galaxy, region, '_'))
            elif region == 'ring' and masks.rings is not None:
                from .rings import beam_ring_jobs
                ring_jobs, records = beam_ring_jobs(config, galaxy)
                jobs.extend(ring_jobs)
                if ring_records is not None:
                    ring_records[galaxy.name] = records
            elif region in {'inner', 'outer', 'ring'}:
                letter = {'inner': 'I', 'outer': 'O', 'ring': 'R'}[region]
                for multiplier in masks.multipliers:
                    number = f'{multiplier:.2f}' if region == 'ring' else str(float(multiplier))
                    suffix = f'_{letter}{number}{masks.radius_tag}'
                    mask_suffix = f'_IO{number}{masks.radius_tag}' if region != 'ring' else suffix
                    jobs.append(StackJob(galaxy, region, suffix, galaxy.cube.parent / f'NGFIT/mask/mask{mask_suffix}.fits', multiplier))
            elif region == 'angle':
                radial_regions = masks.angle_radial_regions or (masks.angle_radial_region,)
                for radial_region in radial_regions:
                    radial_multipliers = masks.multipliers if radial_region else (None,)
                    for multiplier in radial_multipliers:
                        for width in masks.angle_widths:
                            for angle in masks.angles:
                                suffix = f'_W{width:03d}_A{angle:03d}'
                                mask_suffix = suffix
                                if radial_region:
                                    letter = 'I' if radial_region == 'inner' else 'O'
                                    mask_suffix = f'_IO{float(multiplier)}{masks.radius_tag}' + suffix
                                    suffix = f'_{letter}{float(multiplier)}{masks.radius_tag}' + suffix
                                jobs.append(StackJob(galaxy, region, suffix, galaxy.cube.parent / f'NGFIT/mask/mask{mask_suffix}.fits',
                                                     multiplier=multiplier, angle=angle, angle_width=width,
                                                     radial_region=radial_region))
            else:
                raise ValueError(f'Unknown region: {region}')
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError('Selections produce duplicate output suffixes; check multipliers/angles/regions')
    return natsorted(jobs, key=lambda job: job.key)


def validate_catalogs(config, galaxies):
    """Skip absent or invalid optical geometry; reject ambiguous entries."""
    if set(config.masks.regions) <= {'whole'}:
        return galaxies
    import pandas as pd
    accepted = []
    kind = catalog_kind(config.masks.catalog)
    if kind == 'optical':
        from .optical import read_optical_geometry, AmbiguousOpticalGeometryError
        table = pd.read_csv(config.masks.catalog, sep=r'\s+')
        unsupported = set(config.masks.regions) - ({'whole', 'inner', 'outer', 'angle'} | ({'ring'} if config.masks.rings else set()))
        if unsupported:
            raise ValueError('NEDLVS optical catalog does not support legacy ring masks; configure beam rings for ring regions')
        for galaxy in galaxies:
            catalog_name = resolve_catalog_name(galaxy.name, set(table['objname']))
            if not (table['objname'] == catalog_name).any():
                print(f'[skip] {galaxy.name}: no optical catalog entry', flush=True)
                continue
            try:
                read_optical_geometry(config.masks.catalog, galaxy.name, center_only=not config.masks.needs_ellipse)
            except AmbiguousOpticalGeometryError:
                raise
            except ValueError as error:
                print(f'[skip] {galaxy.name}: invalid optical geometry ({error})', flush=True)
                continue
            accepted.append(galaxy)
    else:
        table = pd.read_csv(config.masks.catalog, sep=r'\s+')
        for galaxy in galaxies:
            name = galaxy.name
            if name == 'VCC1091+98':
                name = 'VCC1091'
            rows = table.loc[table['Name'] == name]
            if rows.empty:
                print(f'[skip] {galaxy.name}: no legacy catalog entry', flush=True)
                continue
            if len(rows) != 1:
                raise ValueError(f'{name}: expected one legacy catalog entry, found {len(rows)}')
            accepted.append(galaxy)
    return accepted
