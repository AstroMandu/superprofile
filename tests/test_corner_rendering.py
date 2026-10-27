"""Corner composition keeps diagnostic pixels without temporary PNGs or figures."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xgfit.plotter.plotter import Plotter, paste_corner_figure


class CornerRenderingTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_render_pixels_survive_corner_close_without_png_io(self):
        atlas = plt.figure()
        corner, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color='red')
        number = corner.number
        with patch.object(corner, 'savefig', side_effect=AssertionError('temporary PNG')):
            with patch.object(plt, 'imread', side_effect=AssertionError('PNG read')):
                paste_corner_figure(atlas, corner, [0, 0, 1, 1])
        self.assertFalse(plt.fignum_exists(number))
        pixels = np.asarray(atlas.axes[0].images[0].get_array())
        self.assertEqual(pixels.shape[-1], 4)
        self.assertTrue(np.any(pixels[:, :, 3] == 0))
        self.assertTrue(np.any((pixels[:, :, 0] > pixels[:, :, 1]) & (pixels[:, :, 3] > 0)))
        # The atlas remains renderable after the source canvas has gone away.
        atlas.canvas.draw()

    def test_corner_is_closed_if_composition_fails(self):
        atlas = plt.figure()
        corner = plt.figure()
        number = corner.number
        with patch.object(atlas, 'add_axes', side_effect=RuntimeError('composition failed')):
            with self.assertRaisesRegex(RuntimeError, 'composition failed'):
                paste_corner_figure(atlas, corner, [0, 0, 1, 1])
        self.assertFalse(plt.fignum_exists(number))

    def test_real_resampling_corner_can_be_composed_without_files(self):
        rng = np.random.default_rng(656)
        names = ['V21', 'S21', 'F21', 'S22', 'F22']
        samples = rng.normal([0, 8, 10, 20, 20], [1, 1, 1, 2, 2], (300, 5))
        model = SimpleNamespace(x=np.arange(-40., 41.), y=np.zeros(81),
                                e_y=np.ones(81), names_param=names)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plotter = Plotter(root, 'TEST', '_', [],
                              pd.DataFrame([dict(zip(names, samples.mean(axis=0)))]),
                              pd.DataFrame([{}]), model,
                              resampled=np.column_stack([samples, np.zeros(300)]))
            atlas = plt.figure()
            corner = plotter.makeplot_corner_resample(savefig=False)
            paste_corner_figure(atlas, corner, [0, 0, 1, 1])
            self.assertEqual(plt.get_fignums(), [atlas.number])
            self.assertEqual(list(root.iterdir()), [])
            atlas.savefig(root / 'atlas.png')
            self.assertGreater((root / 'atlas.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
