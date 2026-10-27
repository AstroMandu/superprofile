"""Regression checks for stale tied centers after resampling adoption."""
import ast
from pathlib import Path
import unittest
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'xgfit/xgfit/resampling.py'
tree = ast.parse(PATH.read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'synchronize_tied_centers')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(PATH), 'exec'), namespace)
synchronize = namespace['synchronize_tied_centers']


class TiedCentersTest(unittest.TestCase):
    def test_vcc656_regression(self):
        frame = pd.DataFrame([dict(V1=-1.608171, V21=-2.461322, V22=30.469678)])
        synchronize(frame, ['F21','F22','V21','S21','S22'], 2)
        self.assertEqual(frame.loc[0, 'V22'], frame.loc[0, 'V21'])
        self.assertEqual(frame.loc[0, 'V1'], -1.608171)

    def test_independently_fitted_center_is_preserved(self):
        frame = pd.DataFrame([dict(V1=0., V21=-2., V22=3.)])
        synchronize(frame, ['V21','V22'], 2)
        self.assertEqual(frame.loc[0, 'V22'], 3.)

    def test_three_components_and_fixed_primary(self):
        frame = pd.DataFrame([dict(V1=2., V31=-10., V32=20., V33=30.)])
        synchronize(frame, [], 3)
        self.assertEqual(frame.loc[0, 'V31'], 2.)
        self.assertEqual(frame.loc[0, 'V32'], 2.)
        self.assertEqual(frame.loc[0, 'V33'], 2.)


if __name__ == '__main__':
    unittest.main()
