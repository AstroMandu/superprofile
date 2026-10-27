"""Immutable experiment settings and command-line parsing."""
import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from .catalogs import catalog_kind

DATA_ROOT = Path('/home/mskim/workspace/research/data')
AVID_GALAXIES = (
    'VCC152', 'VCC169', 'VCC334', 'VCC340', 'VCC566', 'VCC613', 'VCC656',
    'VCC667', 'VCC693', 'VCC697', 'VCC699', 'VCC1091', 'VCC1411', 'VCC1992', 'VCC2034',
)


@dataclass(frozen=True)
class FitConfig:
    mode: str = 'HERMOM'
    model: str = '2GFIT'
    resamples: int = 5000
    statistics: str = 'MEDIAN'
    method: str = 'Nelder-Mead'
    initmod: int = 123
    intermediate_plots: bool = False
    correlation_diagnostics: bool = False
    walk_plot_max_steps: int = 1000
    walk_plot_max_walkers: int = 30
    use_secondary_vf: bool = True
    truth_from_resampling: bool = True
    fallback_to_2gfit: bool = False
    guess_from_whole: bool = False
    vdisp_low_intrinsic: float = 0
    classify: bool = True
    overwrite_classify: bool = False
    snlim_peak: int = 10000
    key_classify: int = 10000


@dataclass(frozen=True)
class RingStopConfig:
    mode: str = 'radius'
    radius: str = 'Ropt'
    multiplier: float = 1.0
    min_fraction: float = 0.5
    consecutive_below: int = 1
    include_trigger_ring: bool = False


@dataclass(frozen=True)
class RingConfig:
    beam_fwhm: str = 'mean'
    width_beams: float = 1.0
    start_beams: float = 0.0
    stop: RingStopConfig = field(default_factory=RingStopConfig)
    final_ring: str = 'drop'


@dataclass(frozen=True)
class MaskConfig:
    cleanup_after_run: bool = False
    regions: tuple[str, ...] = ('inner', 'outer')
    catalog: Path = DATA_ROOT / 'catalog/diameters_NEDLVS.csv'
    radius_column: str = 'RHI'
    multipliers: tuple[float, ...] = (1.0,)
    angles: tuple[int, ...] = tuple(range(0, 360, 15))
    angle_widths: tuple[int, ...] = (120,)
    angle_radial_region: str | None = None
    angle_radial_regions: tuple[str, ...] = ()
    ring_width: str = 'beam'
    rings: RingConfig | None = None
    ra_column: str = 'RA'
    dec_column: str = 'Dec'

    @property
    def needs_ellipse(self):
        return (bool(set(self.regions) & {'inner', 'outer', 'ring'})
                or ('angle' in self.regions and bool(
                    self.angle_radial_region or self.angle_radial_regions)))

    @property
    def catalog_kind(self):
        return catalog_kind(self.catalog)

    @property
    def is_optical_catalog(self):
        return self.catalog_kind == 'optical'

    @property
    def radius_tag(self):
        return 'Ropt' if self.is_optical_catalog else self.radius_column


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = DATA_ROOT / 'VIVA/VIVA_test'
    output_dir: Path | None = None
    output_dir_name: str | None = None
    galaxies: tuple[str, ...] | None = None
    cube_name: str = 'cube.fits'
    workers: int = 2
    overwrite: bool = False
    dry_run: bool = False
    progress: str = 'file'
    fit: FitConfig = field(default_factory=FitConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)

    @property
    def output(self):
        if self.output_dir is not None:
            return self.output_dir
        tag = '_Ropt' if self.masks.is_optical_catalog else ''
        return self.data_dir.with_name(f'{self.data_dir.name}_{self.fit.model}_{self.fit.mode}{tag}')

    def output_for(self, galaxy):
        if self.output_dir is not None:
            return self.output_dir
        output = galaxy.cube.parent / 'NGFIT'
        return output / self.output_dir_name if self.output_dir_name is not None else output

    def validate(self):
        if self.output_dir_name is not None and (
                not self.output_dir_name.strip() or self.output_dir_name in {'.', '..'}
                or '/' in self.output_dir_name or '\\' in self.output_dir_name):
            raise ValueError('output_dir_name must be a single non-empty folder name')
        if self.progress not in {'file', 'terminal'}:
            raise ValueError('progress must be file or terminal')
        if self.workers < 1 or self.fit.resamples < 1:
            raise ValueError('workers and resamples must be positive')
        if self.fit.walk_plot_max_steps < 2 or self.fit.walk_plot_max_walkers < 1:
            raise ValueError('walk_plot_max_steps must be >= 2 and walk_plot_max_walkers >= 1')
        if self.fit.mode != 'BAY' and not self.fit.use_secondary_vf:
            raise ValueError('MOM/HER/HERMOM require a secondary velocity field')
        if not self.masks.regions:
            raise ValueError('select at least one region')
        if set(self.masks.regions) - {'whole', 'inner', 'outer', 'ring', 'angle'}:
            raise ValueError(f'unsupported regions: {self.masks.regions}')
        if 'angle' in self.masks.regions and (not self.masks.angles or not self.masks.angle_widths):
            raise ValueError('angular selections require angles and widths')
        if self.masks.angle_radial_region not in {None, 'inner', 'outer'}:
            raise ValueError('angle_radial_region must be null, inner, or outer')
        if set(self.masks.angle_radial_regions) - {'inner', 'outer'}:
            raise ValueError('angle_radial_regions accepts inner and outer')
        if self.masks.angle_radial_regions and self.masks.angle_radial_region is not None:
            raise ValueError('use angle_radial_region or angle_radial_regions, not both')
        if len(set(self.masks.angle_radial_regions)) != len(self.masks.angle_radial_regions):
            raise ValueError('angle_radial_regions must be unique')
        if self.masks.radius_column not in {'RHI', 'r25'}:
            raise ValueError(f'unsupported legacy radius column: {self.masks.radius_column}')
        rings = self.masks.rings
        if rings is not None:
            if 'ring' not in self.masks.regions or not self.masks.is_optical_catalog:
                raise ValueError('masks.rings requires region ring and an optical catalog')
            if rings.beam_fwhm not in {'mean', 'major', 'minor'}:
                raise ValueError('rings.beam_fwhm must be mean, major, or minor')
            if (not math.isfinite(rings.width_beams) or rings.width_beams <= 0
                    or not math.isfinite(rings.start_beams) or rings.start_beams < 0):
                raise ValueError('rings requires positive width_beams and nonnegative start_beams')
            stop = rings.stop
            if stop.mode not in {'radius', 'filling_factor'} or stop.radius not in {'Ropt', 'beam'}:
                raise ValueError('rings.stop requires mode radius/filling_factor and radius Ropt/beam')
            if not math.isfinite(stop.multiplier) or stop.multiplier <= 0:
                raise ValueError('rings.stop.multiplier must be positive')
            if not math.isfinite(stop.min_fraction) or not 0 <= stop.min_fraction <= 1:
                raise ValueError('rings.stop.min_fraction must be in [0, 1]')
            if stop.consecutive_below < 1:
                raise ValueError('rings.stop.consecutive_below must be positive')
            if rings.final_ring not in {'drop', 'clip'}:
                raise ValueError('rings.final_ring must be drop or clip')
            if stop.mode == 'filling_factor' and self.fit.mode == 'BAY':
                raise ValueError('filling_factor rings currently require MOM, HER, or HERMOM')
        if not self.masks.multipliers or any(not math.isfinite(m) or m <= 0 for m in self.masks.multipliers):
            raise ValueError('radius multipliers must be positive')
        if len(set(self.masks.multipliers)) != len(self.masks.multipliers):
            raise ValueError('radius multipliers must be unique')
        if any(not 0 < w <= 360 for w in self.masks.angle_widths):
            raise ValueError('angle widths must lie in (0, 360]')
        if any(not 0 <= a < 360 for a in self.masks.angles):
            raise ValueError('angles must lie in [0, 360)')
        if self.fit.mode not in {'BAY', 'MOM', 'HER', 'HERMOM'}:
            raise ValueError(f'unsupported velocity mode: {self.fit.mode}')
        if self.fit.statistics not in {'MEDIAN', 'MAP'}:
            raise ValueError('statistics must be MEDIAN or MAP')
        if self.fit.initmod not in {1, 2, 3, 12, 13, 123}:
            raise ValueError('unsupported initmod')
        if self.masks.ring_width not in {'beam', 'halfbeam', '2*beam'}:
            raise ValueError('unsupported ring width')
        if self.fit.model not in {'2GFIT', '3GFIT'}:
            raise ValueError(f'unsupported fit model: {self.fit.model}')
        return self


def preset(name):
    if name not in {'viva-test', 'legacy-avid'}:
        raise ValueError(f'Unknown preset: {name}')
    config = PipelineConfig()
    if name == 'legacy-avid':
        return replace(config, data_dir=DATA_ROOT / 'AVID/AVID_nattaper_hann_orig',
                       galaxies=AVID_GALAXIES, workers=60,
                       fit=replace(config.fit, guess_from_whole=True, overwrite_classify=True),
                       masks=replace(config.masks, catalog=DATA_ROOT / 'catalog/cat_diameters.csv',
                                     regions=('angle',), multipliers=(0.6,)))
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plan, stack, and fit galaxy HI superprofiles.')
    parser.add_argument('--config', type=Path, help='YAML experiment file; CLI options override its values')
    parser.add_argument('--progress', choices=('file', 'terminal'), help='Write Plotfit_stat.csv (default) or print live status updates')
    parser.add_argument('--preset', choices=('viva-test', 'legacy-avid'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--output-dir-name', help='Subfolder under each cube directory/NGFIT; --output-dir takes precedence')
    parser.add_argument('--galaxies', nargs='+')
    parser.add_argument('--cube-name')
    parser.add_argument('--catalog', type=Path, help='NEDLVS optical or legacy center/radius table')
    parser.add_argument('--radius-column', choices=('RHI', 'r25'), help='Use legacy ellipse/ring/angle masks')
    parser.add_argument('--regions', nargs='+', choices=('whole', 'inner', 'outer', 'ring', 'angle'))
    parser.add_argument('--multipliers', nargs='+', type=float)
    parser.add_argument('--angles', nargs='+', type=int)
    parser.add_argument('--angle-widths', nargs='+', type=int)
    parser.add_argument('--angle-radial-regions', nargs='+', choices=('inner', 'outer'))
    parser.add_argument('--angle-radial-region', choices=('inner', 'outer'), help='Restrict angular wedges to a radial ellipse')
    parser.add_argument('--ring-width', choices=('beam', 'halfbeam', '2*beam'))
    parser.add_argument('--ra-column')
    parser.add_argument('--dec-column')
    parser.add_argument('--mode', choices=('BAY', 'MOM', 'HER', 'HERMOM'))
    parser.add_argument('--model', choices=('2GFIT', '3GFIT'))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--resamples', type=int)
    parser.add_argument('--statistics', choices=('MEDIAN', 'MAP'))
    parser.add_argument('--method')
    parser.add_argument('--initmod', type=int, choices=(1, 2, 3, 12, 13, 123))
    parser.add_argument('--guess-from-whole', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--classify', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--overwrite-classify', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=None, help='Validate and show jobs without writing files')
    args = parser.parse_args(argv)
    try:
        if args.config:
            from .yaml_config import load_config
            config = load_config(args.config, args.preset)
        else:
            config = preset(args.preset or 'viva-test')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    top = {k: getattr(args, k) for k in ('data_dir', 'output_dir', 'output_dir_name', 'cube_name', 'workers', 'progress') if getattr(args, k) is not None}
    for key in ('data_dir', 'output_dir'):
        if key in top:
            top[key] = top[key].resolve()
    if args.galaxies:
        top['galaxies'] = tuple(args.galaxies)
    elif args.data_dir:
        top['galaxies'] = None
    top.update({key: getattr(args, key) for key in ('overwrite', 'dry_run') if getattr(args, key) is not None})
    fits = {k: getattr(args, k) for k in ('mode', 'model', 'resamples', 'statistics', 'method', 'initmod', 'guess_from_whole', 'classify') if getattr(args, k) is not None}
    if args.overwrite_classify is not None:
        fits['overwrite_classify'] = args.overwrite_classify
    masks = {}
    for key in ('regions', 'multipliers', 'angles', 'angle_widths', 'angle_radial_regions'):
        if getattr(args, key) is not None:
            masks[key] = tuple(dict.fromkeys(getattr(args, key)))
    if args.catalog is not None:
        masks['catalog'] = args.catalog.resolve()
    for key in ('ring_width', 'ra_column', 'dec_column', 'angle_radial_region'):
        if getattr(args, key) is not None:
            masks[key] = getattr(args, key)
    if args.angle_radial_regions is not None:
        masks['angle_radial_region'] = None
    if args.angle_radial_region is not None:
        masks['angle_radial_regions'] = ()
    if args.angle_radial_region is not None and args.angle_radial_regions is not None:
        parser.error('use --angle-radial-region or --angle-radial-regions, not both')
    if args.radius_column:
        masks['radius_column'] = args.radius_column
        if args.catalog is None:
            if config.masks.catalog_kind == 'optical':
                masks['catalog'] = DATA_ROOT / 'catalog/cat_diameters.csv'
        elif catalog_kind(args.catalog) == 'optical':
            parser.error('--radius-column applies only to a legacy catalog')
    config = replace(config, **top, fit=replace(config.fit, **fits), masks=replace(config.masks, **masks))
    try:
        return config.validate()
    except ValueError as error:
        parser.error(str(error))
