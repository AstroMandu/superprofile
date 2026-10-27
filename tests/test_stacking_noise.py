"""Validate summed-flux errors against independent correlated beam blocks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shiftnstack.subroutines_shiftnstack import summed_flux_error


class StackingNoiseTest(unittest.TestCase):
    def test_one_beam_preserves_noise_and_four_beams_double_it(self):
        np.testing.assert_allclose(summed_flux_error(0.0006, [0, 28, 112], 28),
                                   [0, 0.0006, 0.0012])

    def test_correlated_pixel_sum_matches_analytic_error(self):
        rng = np.random.default_rng(531)
        noise = 0.0006
        for pixels_per_beam in (1, 8, 28):
            with self.subTest(pixels_per_beam=pixels_per_beam):
                beams = rng.normal(0, noise, size=(30000, 12))
                pixel_sum = np.repeat(beams, pixels_per_beam, axis=1).sum(axis=1)
                flux = pixel_sum / pixels_per_beam
                expected = summed_flux_error(noise, 12 * pixels_per_beam, pixels_per_beam)
                self.assertAlmostEqual(np.std(flux) / expected, 1, delta=0.02)


if __name__ == '__main__':
    unittest.main()
