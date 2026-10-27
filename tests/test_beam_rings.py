"""Beam scaling, stopping decisions, boundary coverage and YAML behavior."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PipelineConfig, MaskConfig, RingConfig, RingStopConfig, parse_args
from pipeline.models import GalaxyInput
from pipeline.rings import beam_ring_jobs, elliptical_radius
from pipeline.masks import prepare_mask
from pipeline.planning import plan_jobs, validate_catalogs
from pipeline.runner import run_pipeline


class BeamRingTest(unittest.TestCase):
    def test_missing_catalog_entries_are_skipped_but_duplicates_fail(self):
        missing = replace(self.galaxy, name='MISSING')
        self.assertEqual(validate_catalogs(self.config, [missing, self.galaxy]), [self.galaxy])
        with patch('pipeline.runner.discover_galaxies', return_value=[missing]):
            self.assertEqual(run_pipeline(self.config), [])
        valid_catalog = self.catalog.read_text()
        for invalid in ('TEST 180 10 20 0.5 nan', 'TEST 180 10 20 0 30',
                        'TEST 180 10 -1 0.5 30', 'TEST nan 10 20 0.5 30'):
            self.catalog.write_text(valid_catalog.splitlines()[0] + '\n' + invalid + '\n')
            self.assertEqual(validate_catalogs(self.config, [self.galaxy]), [])
        self.catalog.write_text(valid_catalog)
        self.catalog.write_text(self.catalog.read_text() + 'TEST 180 10 20 0.5 30\n')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            validate_catalogs(self.config, [self.galaxy])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        folder = root / 'TEST'
        folder.mkdir()
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [16, 16]
        wcs.wcs.crval = [180, 10]
        wcs.wcs.cdelt = [-1/3600, 1/3600]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        header = wcs.to_header()
        header['BMAJ'] = 4/3600
        header['BMIN'] = 1/3600
        cube = folder / 'cube.fits'
        fits.writeto(cube, np.ones((1,31,31)), header)
        self.catalog = root / 'catalog.csv'
        self.catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 20 0.5 30\n')
        self.galaxy = GalaxyInput('TEST', cube, None, folder / 'hermite.npy', 'HERMOM')
        rings = RingConfig(stop=RingStopConfig(radius='beam', multiplier=2.5))
        self.config = PipelineConfig(masks=MaskConfig(regions=('ring',), catalog=self.catalog, rings=rings))

    def test_radius_drop_clip_and_complementary_masks(self):
        jobs, records = beam_ring_jobs(self.config.validate(), self.galaxy)
        self.assertEqual(len(jobs), 2)
        self.assertAlmostEqual(jobs[0].beam_arcsec, 2)
        self.assertEqual(records[-1]['reason'], 'partial_dropped')
        config = replace(self.config, masks=replace(self.config.masks,
                         rings=replace(self.config.masks.rings, final_ring='clip')))
        jobs, records = beam_ring_jobs(config, self.galaxy)
        self.assertEqual(len(jobs), 3)
        self.assertAlmostEqual(jobs[-1].outer_arcsec, 5)
        cache, union = {}, np.zeros((31,31), dtype=bool)
        for job in jobs:
            prepare_mask(job, config.masks, cache)
            selected = np.isfinite(fits.getdata(job.mask))
            self.assertFalse((union & selected).any())
            union |= selected
        self.assertEqual(len(cache), 1)
        self.assertTrue(union[15,15])

    def test_beam_options_scale_per_galaxy(self):
        for option, expected in [('mean', 2), ('major', 4), ('minor', 1)]:
            config = replace(self.config, masks=replace(self.config.masks,
                             rings=replace(self.config.masks.rings, beam_fwhm=option)))
            jobs, _ = beam_ring_jobs(config, self.galaxy)
            self.assertAlmostEqual(jobs[0].beam_arcsec, expected)
        with fits.open(self.galaxy.cube, mode='update') as hdus:
            del hdus[0].header['BMIN']
        with self.assertRaisesRegex(ValueError, 'BMAJ and BMIN'):
            beam_ring_jobs(self.config, self.galaxy)

    def test_filling_stop_excludes_or_includes_trigger(self):
        stop = RingStopConfig(mode='filling_factor', min_fraction=0.5)
        config = replace(self.config, masks=replace(self.config.masks,
                         rings=replace(self.config.masks.rings, stop=stop)))
        with patch('pipeline.rings.usable_pixels', return_value=np.zeros((31,31), bool)):
            jobs, rows = beam_ring_jobs(config, self.galaxy)
            self.assertEqual(jobs, [])
            self.assertEqual(rows[0]['filling_factor'], 0)
            self.assertFalse(rows[0]['retained'])
            config = replace(config, masks=replace(config.masks, rings=replace(config.masks.rings,
                             stop=replace(stop, include_trigger_ring=True))))
            jobs, rows = beam_ring_jobs(config, self.galaxy)
            self.assertEqual(len(jobs), 0)  # Empty trigger rings cannot be fitted.
            self.assertFalse(rows[0]['retained'])

    def test_zero_threshold_stops_at_zero_usable_pixels(self):
        stop = RingStopConfig(mode='filling_factor', min_fraction=0)
        config = replace(self.config, masks=replace(self.config.masks,
                         rings=replace(self.config.masks.rings, stop=stop))).validate()
        usable = np.zeros((31,31), bool)
        usable[15,15] = True
        with patch('pipeline.rings.usable_pixels', return_value=usable):
            jobs, rows = beam_ring_jobs(config, self.galaxy)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]['filling_factor'], 0)
        self.assertEqual(rows[1]['usable_pixels'], 0)
        self.assertEqual(rows[1]['reason'], 'filling_factor_stop')

    def test_consecutive_low_rings_and_image_boundary_denominator(self):
        stop = RingStopConfig(mode='filling_factor', consecutive_below=2)
        config = replace(self.config, masks=replace(self.config.masks,
                         rings=replace(self.config.masks.rings, stop=stop)))
        from pipeline.optical import read_optical_geometry
        geometry = read_optical_geometry(self.catalog, 'TEST')
        y, x = np.indices((31,31))
        radius = elliptical_radius(WCS(fits.getheader(self.galaxy.cube)).celestial, geometry, x, y)
        usable = (radius >= 2) & (radius < 4)
        with patch('pipeline.rings.usable_pixels', return_value=usable):
            jobs, rows = beam_ring_jobs(config, self.galaxy)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(jobs), 1)  # Empty ring 0 is skipped; low rings 2,3 stop.
        self.assertEqual([row['retained'] for row in rows], [False, True, False, False])
        config = replace(config, masks=replace(config.masks, rings=replace(config.masks.rings,
                         start_beams=7, width_beams=4, stop=replace(stop, consecutive_below=1, min_fraction=1))))
        with patch('pipeline.rings.usable_pixels', return_value=np.ones((31,31), bool)):
            jobs, rows = beam_ring_jobs(config, self.galaxy)
        self.assertEqual(jobs, [])
        self.assertGreater(rows[0]['total_pixels'], rows[0]['selected_pixels'])
        self.assertLess(rows[0]['filling_factor'], 1)

    def test_nested_yaml_and_dry_planning_write_nothing(self):
        config = parse_args(['--config', str(Path(__file__).resolve().parents[1] / 'configs/example_beam_rings.yaml')])
        self.assertEqual(config.masks.rings.beam_fwhm, 'mean')
        self.assertEqual(config.masks.rings.stop.min_fraction, .5)
        records = {}
        jobs = plan_jobs(self.config, [self.galaxy], ring_records=records)
        self.assertEqual(len(jobs), 2)
        self.assertIn('TEST', records)
        self.assertFalse((self.galaxy.cube.parent / 'NGFIT' / 'mask').exists())
        for option in ('geometric_mean', 'invalid'):
            with self.assertRaisesRegex(ValueError, 'beam_fwhm'):
                replace(self.config, masks=replace(self.config.masks,
                        rings=replace(self.config.masks.rings, beam_fwhm=option))).validate()

    def test_filling_threshold_is_strict(self):
        rings = replace(self.config.masks.rings, stop=RingStopConfig(mode='filling_factor'))
        config = replace(self.config, masks=replace(self.config.masks, rings=rings))
        usable = np.ones((31,31), bool)
        usable[15,15] = False
        with patch('pipeline.rings.usable_pixels', return_value=usable):
            _, rows = beam_ring_jobs(config, self.galaxy)
            fraction = rows[0]['filling_factor']
            self.assertGreater(fraction, 0)
            config = replace(config, masks=replace(config.masks, rings=replace(rings,
                             stop=replace(rings.stop, min_fraction=fraction))))
            _, rows = beam_ring_jobs(config, self.galaxy)
        self.assertTrue(rows[0]['retained'])

    def test_runner_saves_geometry_and_resumes_ring_results(self):
        self.galaxy.secondary_vf.touch()
        config = replace(self.config, data_dir=self.galaxy.cube.parent.parent,
                         workers=1, output_dir_name='test_rings')
        seen = []

        def fake_fit(task):
            self.assertTrue(task.stack.mask.exists())
            seen.append(task.stack.key)
            frame = pd.DataFrame({'Name': [task.stack.galaxy.name], 'suffix': [task.stack.suffix]})
            for kind in ('gfit', 'para'):
                frame.to_string(task.temp / f'{len(seen)}_plotfitter_temp_{kind}.csv', index=False)
            return task.stack.key

        with patch('pipeline.worker.fit_stack', side_effect=fake_fit):
            run_pipeline(config)
            output = config.output_for(self.galaxy)
            table = pd.read_csv(output / 'ring_geometry.csv')
            self.assertEqual(len(table), 3)
            self.assertEqual(table['retained'].tolist(), [True, True, False])
            self.assertTrue((output / 'info_stacked_GFIT.csv').exists())
            run_pipeline(config)
            self.assertEqual(len(seen), 2)


if __name__ == '__main__':
    unittest.main()
