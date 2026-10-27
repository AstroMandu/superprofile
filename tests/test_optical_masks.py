"""Check radius-vs-diameter, PA convention, and complementary FITS masks."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optical_masks import create_optical_masks, read_optical_geometry


class OpticalMasksTest(unittest.TestCase):
    def test_ellipse_geometry_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            galaxy = root / 'TEST'
            galaxy.mkdir()
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [16, 16]
            wcs.wcs.crval = [180, 10]
            wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cube = galaxy / 'cube.fits'
            wcs.wcs.radesys = 'FK5'
            wcs.wcs.equinox = 2000
            fits.writeto(cube, np.zeros((31, 31)), wcs.to_header())
            catalog = root / 'diameters.csv'
            catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 20 0.5 0\n')
            masks = create_optical_masks(cube, catalog)
            inner = np.isfinite(fits.getdata(masks['I']))
            outer = np.isfinite(fits.getdata(masks['O']))
            self.assertTrue(np.all(inner ^ outer))
            self.assertTrue(inner[15, 15])
            self.assertTrue(inner[24, 15])  # north: semi-major radius is 10
            self.assertFalse(inner[26, 15])
            self.assertTrue(inner[15, 19])  # east/west: semi-minor radius is 5
            self.assertFalse(inner[15, 21])
            self.assertEqual(fits.getheader(masks['I'])['ROPT'], 10)
            smaller = create_optical_masks(cube, catalog, multiplier=0.5)
            self.assertEqual(smaller['I'].name, 'mask_I0.5Ropt.fits')
            self.assertEqual(fits.getheader(smaller['I'])['ROPT'], 5)
            self.assertLess(np.isfinite(fits.getdata(smaller['I'])).sum(), inner.sum())
            catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 20 0.5 90\n')
            rotated = create_optical_masks(cube, catalog)
            mask90 = np.isfinite(fits.getdata(rotated['I']))
            self.assertTrue(mask90[15, 24])
            self.assertFalse(mask90[21, 15])
            catalog.write_text('objname Diam_ra Diam_dec Diam Diam_ba Diam_pa\nTEST 180 10 NaN 0.5 0\n')
            with self.assertRaises(ValueError):
                read_optical_geometry(catalog, 'TEST')
            with self.assertRaises(ValueError):
                read_optical_geometry(catalog, 'MISSING')


if __name__ == '__main__':
    unittest.main()
