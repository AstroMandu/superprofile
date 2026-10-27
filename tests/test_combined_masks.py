"""Angular/radial selection is an intersection and gets a distinct result key."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import MaskConfig, PipelineConfig
from pipeline.masks import prepare_mask, write_geometry
from pipeline.models import GalaxyInput
from pipeline.planning import plan_jobs, validate_catalogs
from shiftnstack.subroutines_shiftnstack import select_spatial_mask
from shiftnstack.class_ShiftnStack_baygaudpi import ShiftnStack


class CombinedMaskTest(unittest.TestCase):
    def test_io_selection_excludes_nan_for_both_regions(self):
        data = np.array([[1., 0., np.nan], [np.nan, 0., 1.]])
        inner = np.isfinite(select_spatial_mask(data, 1))
        outer = np.isfinite(select_spatial_mask(data, 0))
        self.assertFalse(np.any(inner & outer))
        np.testing.assert_array_equal(inner | outer, np.isfinite(data))
        np.testing.assert_array_equal(inner, data == 1)
        np.testing.assert_array_equal(outer, data == 0)
        np.testing.assert_array_equal(select_spatial_mask(data), data)
        with self.assertRaises(ValueError):
            select_spatial_mask(data, 2)

    def test_baygaud_branch_honors_region_before_consuming_components(self):
        for value in (0, 1):
            with self.subTest(value=value):
                stack = ShiftnStack.__new__(ShiftnStack)
                stack.path_clfy = Path('/unused')
                stack.path_mask = Path('/unused/shared.fits')
                stack.bgsub = stack.pbar = False
                stack.map_stack_method = np.zeros((1, 3))
                stack._log_pixel_records = []
                stack.writestat = Mock()
                source = np.array([[1., 0., np.nan]])
                selection = select_spatial_mask(source, value)
                # Empty component records isolate spatial eligibility from fitting.
                records = {f'{x},0': {} for x in range(3)}
                with patch('shiftnstack.class_ShiftnStack_baygaudpi.read_ngfits', return_value=records):
                    stack._stack_baygaud(np.zeros((2, 1, 3)), selection,
                                        np.zeros(2), np.zeros(2), np.zeros(2), Mock(), 0, 0)
                np.testing.assert_array_equal(stack.map_stack_method == 1, source == value)

    def test_center_only_catalog_allows_unrestricted_wedges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'TEST'
            folder.mkdir()
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [16, 16]
            wcs.wcs.crval = [180, 10]
            wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cube = folder / 'cube.fits'
            fits.writeto(cube, np.ones((31, 31)), wcs.to_header())
            catalog = root / 'catalog.csv'
            galaxy = GalaxyInput('TEST', cube, None, None, 'MOM')
            masks = MaskConfig(catalog=catalog, regions=('whole', 'angle'),
                               angles=(0,), angle_widths=(90,))
            config = PipelineConfig(masks=masks)
            for content in (
                'objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 NaN NaN NaN\n',
                'objname Diam_ra Diam_dec\nTEST 180 10\n',
            ):
                with self.subTest(catalog=content):
                    catalog.write_text(content)
                    self.assertEqual(validate_catalogs(config, [galaxy]), [galaxy])
                    write_geometry(masks, [galaxy], root)
                    self.assertTrue((root / "optical_geometry.csv").is_file())
                    job = next(job for job in plan_jobs(config, [galaxy]) if job.region == 'angle')
                    for cache in (None, {}):
                        prepare_mask(job, masks, cache=cache)
                        wedge = np.isfinite(fits.getdata(job.mask))
                        self.assertTrue(wedge[25, 15])
                        self.assertFalse(wedge[5, 15])
                    for regions, radial in ((('inner',), None), (('outer',), None),
                                            (('angle',), 'inner'), (('angle',), 'outer')):
                        restricted = replace(config, masks=replace(masks, regions=regions,
                                                                  angle_radial_region=radial))
                        self.assertEqual(validate_catalogs(restricted, [galaxy]), [])
            catalog.write_text('objname Diam_ra Diam_dec\nTEST NaN 10\n')
            self.assertEqual(validate_catalogs(config, [galaxy]), [])
            catalog.write_text('objname Diam_ra Diam_dec\nTEST 180 10\nTEST 180 10\n')
            with self.assertRaises(ValueError):
                validate_catalogs(config, [galaxy])

    def test_both_radial_regions_queue_distinct_jobs(self):
        galaxy = GalaxyInput('TEST', Path('/tmp/TEST/cube.fits'), None, None, 'MOM')
        masks = MaskConfig(catalog=Path('/home/mskim/workspace/research/data/catalog/cat_diameters.csv'),
                   regions=('angle',), radius_column='r25',
                           angles=(0, 90), angle_widths=(120,), angle_radial_regions=('inner', 'outer'))
        config = PipelineConfig(masks=masks).validate()
        jobs = plan_jobs(config, [galaxy])
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.suffix for job in jobs},
                         {'_I1.0r25_W120_A000', '_I1.0r25_W120_A090',
                          '_O1.0r25_W120_A000', '_O1.0r25_W120_A090'})
        self.assertEqual({job.radial_region for job in jobs}, {'inner', 'outer'})
        self.assertEqual(len({job.mask for job in jobs}), 2)
        self.assertEqual({job.mask_value for job in jobs}, {0, 1})
        self.assertTrue(all(job.mask.name.startswith('mask_IO') for job in jobs))

    def test_nedlvs_wedges_use_optical_center_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'TEST'
            folder.mkdir()
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [16, 16]
            wcs.wcs.crval = [180, 10]
            wcs.wcs.cdelt = [-1/3600, 1/3600]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            wcs.wcs.radesys = 'FK5'
            wcs.wcs.equinox = 2000
            cube = folder / 'cube.fits'
            fits.writeto(cube, np.ones((2, 31, 31)), wcs.to_header())
            catalog = root / 'nedlvs.csv'
            catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 20 0.5 30\n')
            galaxy = GalaxyInput('TEST', cube, None, None, 'MOM')
            masks = MaskConfig(catalog=catalog, regions=('angle',), angles=(0,),
                               angle_widths=(120,), angle_radial_regions=('inner', 'outer'))
            config = PipelineConfig(masks=masks).validate()
            jobs = plan_jobs(config, [galaxy])
            self.assertEqual(jobs[0].mask, jobs[1].mask)
            combined = {}
            for job in jobs:
                prepare_mask(job, masks)
                combined[job.radial_region] = np.isfinite(select_spatial_mask(fits.getdata(job.mask), job.mask_value))
                self.assertIn('Ropt', job.suffix)
                self.assertEqual(fits.getheader(job.mask)['ROPT'], 10)
            cache = {}
            original = WCS.pixel_to_world
            with patch.object(WCS, 'pixel_to_world', autospec=True, side_effect=original) as transform, \
                 patch('pipeline.optical.fits.writeto', wraps=fits.writeto) as write:
                for job in jobs:
                    prepare_mask(job, masks, cache=cache)
                    np.testing.assert_array_equal(np.isfinite(select_spatial_mask(fits.getdata(job.mask), job.mask_value)),
                                                  combined[job.radial_region])
                # Inner and outer share one expensive sky-coordinate conversion.
                self.assertEqual(transform.call_count, 1)
                self.assertEqual(write.call_count, 2)  # One radial IO + one angular IO.
            values = fits.getdata(jobs[0].mask)
            self.assertEqual(set(values[np.isfinite(values)]), {0, 1})
            self.assertEqual(fits.getheader(jobs[0].mask)['REGION'], 'IO')
            self.assertFalse((folder / 'NGFIT/mask/mask_I1.0Ropt.fits').exists())
            self.assertFalse((folder / 'NGFIT/mask/mask_O1.0Ropt.fits').exists())
            free_config = replace(config, masks=replace(masks, angle_radial_regions=()))
            free_job = plan_jobs(free_config, [galaxy])[0]
            prepare_mask(free_job, free_config.masks)
            wedge = np.isfinite(fits.getdata(free_job.mask))
            self.assertEqual(free_job.suffix, '_W120_A000')
            self.assertTrue(wedge[25, 15])  # image north is angle zero
            self.assertFalse(wedge[5, 15])
            self.assertFalse(np.any(combined['inner'] & combined['outer']))
            np.testing.assert_array_equal(wedge, combined['inner'] | combined['outer'])
            ellipse = fits.getdata(folder / 'NGFIT/mask/mask_IO1.0Ropt.fits') == 1
            np.testing.assert_array_equal(combined['inner'], wedge & ellipse)

    def test_inner_and_outer_wedges_partition_an_unrestricted_wedge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            galaxy_dir = root / 'TEST'
            galaxy_dir.mkdir()
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [16, 16]
            wcs.wcs.crval = [180, 10]
            wcs.wcs.cdelt = [-1/3600, 1/3600]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cube = galaxy_dir / 'cube.fits'
            fits.writeto(cube, np.ones((2, 31, 31)), wcs.to_header())
            catalog = root / 'centers.csv'
            catalog.write_text('Name RA Dec r25 b/a PA\nTEST 180 10 10 0.5 30\n')
            galaxy = GalaxyInput('TEST', cube, None, None, 'MOM')
            masks = MaskConfig(catalog=catalog, regions=('angle',),
                               radius_column='r25', angles=(0,), angle_widths=(120,))
            selections = {}
            for radial_region in (None, 'inner', 'outer'):
                config = PipelineConfig(masks=replace(masks, angle_radial_region=radial_region))
                job = plan_jobs(config, [galaxy])[0]
                prepare_mask(job, config.masks)
                selections[radial_region] = np.isfinite(select_spatial_mask(fits.getdata(job.mask), job.mask_value))
                if radial_region == 'inner':
                    self.assertEqual(job.suffix, '_I1.0r25_W120_A000')
                    ellipse = fits.getdata(galaxy_dir / 'NGFIT/mask/mask_IO1.0r25.fits') == 1
                    np.testing.assert_array_equal(selections['inner'], selections[None] & ellipse)
            self.assertFalse(np.any(selections['inner'] & selections['outer']))
            np.testing.assert_array_equal(selections[None], selections['inner'] | selections['outer'])
            self.assertGreater(selections['inner'].sum(), 0)
            self.assertGreater(selections['outer'].sum(), 0)
            # A fresh preparation cache writes the shared radial/wedge pair once.
            config = PipelineConfig(masks=replace(masks, angle_radial_regions=('inner', 'outer')))
            jobs = plan_jobs(config, [galaxy])
            self.assertEqual(jobs[0].mask, jobs[1].mask)
            cache = {}
            with patch('run_makemask_ellipse.fits.writeto', wraps=fits.writeto) as write:
                for job in jobs:
                    prepare_mask(job, config.masks, cache=cache)
                self.assertEqual(write.call_count, 2)
            values = fits.getdata(jobs[0].mask)
            np.testing.assert_array_equal(np.isfinite(values), selections[None])
            self.assertTrue(np.isnan(values[~selections[None]]).all())
            for job in jobs:
                np.testing.assert_array_equal(
                    np.isfinite(select_spatial_mask(values, job.mask_value)),
                    selections[job.radial_region])


if __name__ == '__main__':
    unittest.main()
