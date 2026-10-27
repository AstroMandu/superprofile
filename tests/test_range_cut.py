"""Exercise preliminary-fit selection without optional plotting dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


def load_methods(namespace):
    path = Path(__file__).resolve().parents[1] / 'xgfit/xgfit/fitting.py'
    # Only parse the relevant class section (also supports Python < 3.12).
    tree = ast.parse(path.read_text().split('    # Residuals')[0])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'makefit_1G_minimize', 'limit_range'}]
    module = ast.parse('from __future__ import annotations')
    module.body.extend(methods)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class RangeCutTest(unittest.TestCase):
    def test_best_converged_start_is_used(self):
        class Model:
            def __init__(self, *args):
                pass

            def make_log_prob_1G(self):
                return None, None

            def array_to_dict_guess(self, values):
                return {'S1': values[2]}

        def optimize(fun, guess, **kwargs):
            width = guess[2]
            return SimpleNamespace(success=width != 5, fun=abs(width - 10), x=guess)

        ns = load_methods({'np': np, 'Gmodel': Model, 'minimize': optimize,
                           'map_params': lambda values, *args, **kwargs: values})
        obj = SimpleNamespace(fit_params_2G={'V21': 'free'}, y=np.ones(20),
                              chansep=1, dispmin=1, dispmax=999, ymax=1,
                              method_minimize='Nelder-Mead')
        method = ns['makefit_1G_minimize']
        self.assertEqual(method(obj, np.arange(20), obj.y, obj.y)['S1'], 10)
        ns['minimize'] = lambda *args, **kwargs: SimpleNamespace(success=False, fun=0)
        with self.assertRaisesRegex(RuntimeError, 'refusing to trim'):
            method(obj, np.arange(20), obj.y, obj.y)

    def test_range_reuses_initial_fit_and_keeps_arrays_aligned(self):
        ns = load_methods({'np': np})
        frame = pd.DataFrame({'x': np.arange(-20, 21), 'y': np.ones(41),
                              'e_y': np.linspace(1, 2, 41)})
        obj = SimpleNamespace(df_stacked=frame, xmin=-20, xmax=20,
                              writestat=Mock(), makefit_1G_minimize=Mock())
        ns['limit_range'](obj, multiplier_disp=3, initial_fit={'V1': 1, 'S1': 2})
        obj.makefit_1G_minimize.assert_not_called()
        np.testing.assert_array_equal(obj.x, np.arange(-5, 8))
        np.testing.assert_array_equal(obj.e_y, frame.loc[frame.x.between(-5, 7), 'e_y'])


if __name__ == '__main__':
    unittest.main()
