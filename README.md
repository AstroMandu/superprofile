Superprofile pipeline
=====================

Run with Python 3.13 from this directory. `pipe_superprofile.py` is the entry
point; experiments are configured through CLI options or immutable Python
objects. No shared global configuration dictionaries are used.

The default experiment is **VIVA_test, HERMOM, two-Gaussian fits, optical inner
and outer masks**, with two workers and up to 5,000 resamples. The fitter can
stop resampling early when its existing convergence criterion is met.

```sh
# Review inputs and pending/completed selections without writing anything.
python3.13 pipe_superprofile.py --dry-run

# Run the default VIVA optical experiment; completed selections are skipped.
python3.13 pipe_superprofile.py

# Choose another sample, output location, or velocity-field method.
python3.13 pipe_superprofile.py \
  --data-dir /path/to/sample \
  --output-dir /path/to/results \
  --catalog /path/to/diameters_NEDLVS.csv \
  --regions inner outer --mode HERMOM --workers 2

# Compare multiple optical radii.
python3.13 pipe_superprofile.py --multipliers 0.5 1.0 1.5

# Use a completed whole-galaxy fit to initialize regional fits.
# Each galaxy's whole fit is committed before its regional workers start.
python3.13 pipe_superprofile.py \
  --regions whole inner outer --guess-from-whole
```

`--workers 1` runs serially. `--galaxies VCC307 ...` restricts discovery, and
`--resamples N`, `--model 2GFIT|3GFIT`, `--statistics MEDIAN|MAP`, `--initmod`,
and `--method` control fitting. Run `--help` for all options. Galaxies missing
their cube, required velocity field, or BAYGAUD inputs are skipped with a
message; catalog entries without usable geometry are also skipped.

YAML experiment files
---------------------

Edit `configs/viva_angles.yaml` to configure the VIVA angular experiment, or
`configs/viva_optical.yaml` for optical inner/outer stacks:

```sh
python3.13 pipe_superprofile.py --config configs/viva_angles.yaml --dry-run
python3.13 pipe_superprofile.py --config configs/viva_angles.yaml
```

YAML uses the field names in `PipelineConfig`, with nested `fit:` and `masks:`
sections. Omitted fields inherit the selected preset (`viva-test` by default).
Angles accept an explicit list or an integer range with an exclusive stop:

```yaml
masks:
  angles: {start: 0, stop: 360, step: 15}  # Same values as np.arange(0, 360, 15)
```

YAML does not evaluate Python expressions. The range expands to 0, 15, …, 345.
Paths are resolved relative to the YAML file, including output paths. Set one
`masks.catalog` path; its header selects NEDLVS optical geometry or the legacy
`Name`/center/radius catalog format. Unknown keys and incorrect types raise errors.

Precedence is **preset → YAML → explicit CLI options**. For example,
`--workers 1` overrides the file, and `--no-overwrite` overrides
`overwrite: true`. PyYAML is required for `--config`; it is installed in the
current Python 3.13 environment. Loading YAML uses the safe loader.

Use `masks.cleanup_after_run: true` to remove selection masks after all pending
fits finish and both result tables are saved. The YAML examples enable this;
set it to `false` to keep masks (the default when the option is omitted).
Cleanup includes used masks and companion masks generated during preparation,
but preserves unrelated older masks. Failed runs, dry runs, and runs with no
pending fits do not remove masks. Input cube/SoFiA masks are not cleanup targets.

Plotting defaults to a final atlas only. Configure the display under `fit:`:

```yaml
fit:
  intermediate_plots: false
  walk_plot_max_steps: 1000
  walk_plot_max_walkers: 30
```

Set `intermediate_plots: true` to also render intermediate fit atlases. A fit
that terminates after 1G still saves its 1G atlas. On an exception, the last
completed fit is rendered as `Plotfit_atlas_<name><suffix>_error.png` when such
a fit exists; the original exception is then raised. Failure before any valid
fit cannot produce a fitted-model atlas. Walker limits affect display only,
preserve the original iteration axis, and do not thin inference samples.

`fit.correlation_diagnostics` defaults to `false`. This skips robust pairwise
trimming and Spearman correlation/p-value calculations, which are not used by
the active fitter's optimization, convergence or reliability checks. The
`rs_*` and `p_rs_*` result columns remain present with `NaN` values. Set this
option to `true` to restore the original robust diagnostics. It does not turn
off corner plots or change the fit samples.

Likelihoods use a shared cached Numba kernel with task data passed as runtime
arguments. Parallel runs prepare this cache in a short-lived subprocess before
launching fit workers. `maxtasksperchild=1` remains enabled for memory isolation;
workers still exit after every task. Changing the imported likelihood equations
invalidates the shared kernel cache through a source fingerprint. The first run
after an equation change may therefore take longer to prepare.

Residual noise is replayed in batches of at most 128 spectra rather than a full
`resamples × channels` matrix, preserving the legacy draws and global RNG order.
Resampling indexes the chain directly without a flattened copy. Logger handlers,
plotting resources, checkpoints and fitter arrays are released on completion or
failure. These measures reduce avoidable retention and peak memory; they do not
guarantee that third-party libraries cannot leak memory.

Legacy selections
-----------------

The previous AVID configuration is retained as an explicit preset. It uses the
previous named galaxy list, HERMOM, 120-degree angular masks every 15 degrees,
and whole-fit initialization. Existing whole results are required unless
`whole` is included in `--regions` or `--no-guess-from-whole` is supplied.

```sh
python3.13 pipe_superprofile.py --preset legacy-avid --dry-run

# Legacy radius table: inner/outer/ring/angle selections are independent.
python3.13 pipe_superprofile.py \
  --data-dir /path/to/sample \
  --catalog /path/to/cat_diameters.csv --radius-column RHI \
  --regions whole inner outer ring angle --multipliers 0.6 \
  --angles 0 90 180 270 --angle-widths 120 --ring-width beam
```

`--radius-column RHI|r25` selects the legacy radius column; when no `--catalog`
is supplied, it also selects the default legacy catalog. `--ra-column` and
`--dec-column` select legacy center columns.
Angular selections retain the existing image-plane convention in
`makemask_angle.py`. Rings retain the existing beam-width convention.

Angular wedges restricted to an ellipse
--------------------------------------

Use `regions: [angle]` with `angle_radial_region: inner` to intersect every
wedge with an inner ellipse, rather than enqueue separate angular and inner
stacks. `outer` restricts wedges to the complement. `null` (the default)
retains unrestricted wedges. `multipliers` controls the radial cut(s).

```yaml
masks:
  regions: [angle]
  catalog: /home/mskim/workspace/research/data/catalog/cat_diameters.csv
  radius_column: r25
  angle_radial_region: inner
  multipliers: [1.0]
  angles: [0, 90, 180, 270]
  angle_widths: [120]
```

To queue **both** radial selections, replace the singular setting with:

```yaml
masks:
  regions: [angle]
  angle_radial_regions: [inner, outer]
```

Use `configs/viva_angles_inner_outer_r25.yaml` for the complete example. Each
angle produces an inner job and an outer job. They share the worker pool:
`workers: 2` runs up to two jobs concurrently and queues the remainder. This
does not impose an inner-first barrier. Completed jobs are skipped on rerun.
Do not set both the singular and plural options in the same file.

Run the supplied `configs/viva_angles_inner_r25.yaml` for inner only. Combined selections have
suffixes such as `_I1.0r25_W120_A000`, so they do not reuse unrestricted angular
results. Listing `regions: [angle, inner]` creates separate stacks; the
`angle_radial_region` setting supplies the intersection. Combined angular masks
accept either catalog format; the catalog header selects the geometry path.
With NEDLVS, both the wedge center and radial ellipse use `Diam_ra`, `Diam_dec`;
the ellipse also uses `Diam / 2`, `Diam_ba`, and `Diam_pa`. Output suffixes
use `Ropt` to distinguish these from legacy `r25` masks. The supplied
`viva_angles_inner_outer_r25.yaml` now selects NEDLVS despite its historical filename.

Inner and outer jobs share a single `mask_IO1.0Ropt.fits` (or `r25`/`RHI`)
per radius. Directional pairs share files such as
`mask_IO1.0Ropt_W120_A000.fits`: inner pixels are 1, outer pixels are 0,
and all pixels outside the angular wedge are NaN. Stacking selects exactly
1 for inner or 0 for outer; NaNs are excluded in both cases. Output suffixes
remain `_I...` and `_O...`. The shared files are generated once per run and
follow `masks.cleanup_after_run` like other masks.

For VCC307, the legacy catalog has `r25 = 150.36` arcsec but its `PA` is NaN.
The existing ellipse helper substitutes 0 degrees for a missing PA. Set a
suitable PA in your chosen catalog before using these ellipses for analysis.
The angular wedge retains the existing image-plane angle convention.

Geometry and numerical behavior
-------------------------------

For NEDLVS masks, the optical semi-major radius is `Diam / 2` arcseconds,
multiplied by `--multipliers`. The ellipse uses `Diam_ra`, `Diam_dec`,
`Diam_ba`, and `Diam_pa` (east of north). FITS sky coordinates are transformed
to ICRS before computing the ellipse. Inner includes its boundary; outer is
its complement across the cube footprint. Shared IO files encode the two
regions as 1 and 0, with NaN outside the valid selection. ShiftnStack converts
the chosen region to a 1/NaN mask and applies its existing HI and velocity-validity masks.
These are the catalog's survey-specific diameters, not uniformly B-band R25.

HERMOM retains Hermite velocities with moment-map fallback. The numerical
`shiftnstack/` and `xgfit/` implementations are unchanged. Galaxy-specific
velocity corrections remain in `pipeline/galaxy_settings.py`; the old spectral
exceptions remain disabled, matching previous behavior.

Outputs and reruns
------------------

Beam-spaced rings are configured in `configs/example_beam_rings.yaml`, with
per-galaxy fit scheduling: when whole initialization is enabled, saving a
galaxy's whole fit releases its regional fits without waiting for other
galaxies' whole fits. Existing completed whole fits can be reused immediately.
The worker limit applies across both whole and regional fits.
The example includes
comments listing every ring option. Set `masks.regions: [whole, ring]` and
`masks.rings` to enable this mode; omit `masks.rings` for the legacy rings.
Optical beam rings use NEDLVS center, position angle and axis ratio, with
adjacent semi-major-axis intervals measured in the selected beam FWHM.
`mean` means `sqrt(BMAJ * BMIN)`; `major` and `minor` select the larger/smaller
FITS beam axis. A beam-width interval does not guarantee independent rings.

Choose `rings.stop.mode: radius` for an outer limit in `Ropt` or beam units,
or `filling_factor` to stop below a chosen spatial coverage fraction. Filling
also stops at zero usable pixels when `min_fraction: 0` (subject to
`consecutive_below`). Empty rings are never fitted.
Filling
uses the same finite velocity and dispersion rules as secondary stacking,
including HERMOM fallback; there is no extra moment-0 detection threshold.
Outside-image annulus pixels count as unfilled. This mode supports MOM, HER
and HERMOM, and reads the velocity inputs even during dry runs.

`consecutive_below` counts consecutive sub-threshold rings; isolated low-fill
rings remain if the sequence recovers. By default the terminal low-fill
sequence is excluded. `include_trigger_ring: true` retains only its last ring
(unless empty). `ring_geometry.csv` records evaluated radii, beam size, pixel
counts, fractions and decisions, including rejected rings. Radius mode leaves
usable pixel counts/fractions blank. These settings replace `multipliers` and
legacy `ring_width` for ring selection. Use a separate `output_dir_name` when
changing beam or stopping settings to keep experiments separate.

By default, each galaxy's results go to `NGFIT/` beside its cube, for example
`VIVA_test/VCC307/NGFIT/`. Each folder contains that galaxy's result tables,
stacked spectra, plots, geometry, and run metadata. Whole-fit initialization
and resume checks read those local tables. `--output-dir` (or YAML `output_dir`)
overrides this with a shared output directory. Existing results in the previous
sibling output directory are not moved or automatically reused.

Set `output_dir_name: 2GFIT_HERMOM_Ropt` in YAML (or pass
`--output-dir-name 2GFIT_HERMOM_Ropt`) to store results in
`<cube directory>/NGFIT/2GFIT_HERMOM_Ropt/`. The name is a single folder name;
omitting it or setting it to null writes directly into `NGFIT/`.

- `<galaxy>_<region>_stacked.csv`: separate stacked spectra for every selection.
- `info_stacked_GFIT.csv`: combined fit results, keyed by `Name` and `suffix`.
- `info_stacked_GFIT_params.csv`: combined derived parameters.
- `optical_geometry.csv`: unscaled catalog ellipse and optical radius.
- `png/`: fitter plots; `Plotfit_stat.csv`: current run's progress.
- `last_run.json`: exact configuration and planned jobs of the last successful run.

A selection is skipped only when its key exists in **both** result tables.
Use a new output directory for a different experiment, or `--overwrite` to
recompute matching selections. Existing result rows are identified by galaxy
and suffix; changing catalog contents or fitter options does not automatically
invalidate a previous result. `--overwrite-classify` separately controls
BAYGAUD classification replacement.

Each run gets a unique `temp_*/` directory. Workers publish fragments
atomically. The parent merges and deduplicates tables before removing the
fragments. Successful runs remove their own temporary directory; failures
retain intermediate files and `run.json` for inspection. Existing directories
from failed runs are not deleted. Complete result rows survive a later failure.
Dry runs neither create output directories nor regenerate masks.

Code map
--------

| File | Responsibility |
| --- | --- |
| `pipe_superprofile.py` | CLI entry point and early runtime setup |
| `pipeline/config.py` | Immutable fit, mask, and experiment settings; presets; CLI |
| `pipeline/yaml_config.py` | Safe YAML loading, type checks, and file-relative paths |
| `pipeline/models.py` | Galaxy inputs, spatial stack jobs, executable fit jobs |
| `pipeline/planning.py` | Input/catalog validation and deterministic job planning |
| `pipeline/optical.py` | NEDLVS ellipse geometry and complementary FITS masks |
| `pipeline/masks.py` | Dispatch to optical or legacy mask generators |
| `pipeline/execution.py` | Serial/process execution and BAYGAUD classification |
| `pipeline/worker.py` | One stack and fit, using the existing numerical engines |
| `pipeline/initialization.py` | Whole-galaxy preguess ratios |
| `pipeline/results.py` | Atomic result tables, resume keys, progress and manifests |
| `pipeline/runner.py` | Experiment lifecycle and whole/regional dependencies |
| `pipeline/runtime.py` | Thread limits and worker environment |
| `pipeline/galaxy_settings.py` | Existing galaxy-specific exceptions |

`pipeline_config.py`, `optical_masks.py`, and `stack_settings.py` retain public
imports for scripts using those module names. The old mutable globals and
`mask_jobs.py` orchestration have been removed. The legacy mask helper scripts
remain because other analysis modules import them.

Python API
----------

```python
from dataclasses import replace
from pathlib import Path
from pipeline.runtime import configure_runtime
configure_runtime()  # before importing numerical libraries
from pipeline.config import PipelineConfig
from pipeline.runner import run_pipeline

config = PipelineConfig()
config = replace(config, data_dir=Path('/path/to/sample'), workers=1,
                 masks=replace(config.masks, multipliers=(0.5, 1.0)))
run_pipeline(config)
```

Protect calls using multiple workers with `if __name__ == '__main__':` in a
Python script. Configuration and job objects are picklable; workers receive
all settings explicitly.

Validation
----------

```sh
python3.13 -m unittest discover -s tests
```

Tests cover optical diameter/radius and PA conventions, FK5/ICRS handling,
complementary masks, job selection, whole-fit dependencies, dry-run behavior,
missing velocity inputs, and result recovery after a simulated disk failure.
