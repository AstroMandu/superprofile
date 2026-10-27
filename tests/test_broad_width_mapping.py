"""Check broad-width mappings against independent physical Gaussian models."""
import ast
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Load the actual numerical package without importing the plotting stack.
ROOT = Path(__file__).resolve().parents[1] / 'xgfit'
PACKAGE = '_broad_width_mapping_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
helpers = types.ModuleType(PACKAGE + '.subroutines')
helpers.np = np
nodes = [n for n in ast.parse((ROOT / 'subroutines.py').read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in {'idx', 'bound_to_span'}]
for node in nodes:
    node.decorator_list = []
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'subroutines.py'), 'exec'), helpers.__dict__)
sys.modules[helpers.__name__] = helpers
from importlib import import_module
numerics = import_module(PACKAGE + '.gmodel')
Gmodel, map_params = numerics.Gmodel, numerics.map_params


class BroadWidthMappingTest(unittest.TestCase):
    def test_all_active_likelihood_variants(self):
        variants = [
            (['F21', 'F22', 'V21', 'S21', 'S22'], 'make_log_prob_2G_V22xB2x'),
            (['F21', 'F22', 'V21', 'S21', 'S22', 'B2'], 'make_log_prob_2G_V22x'),
            (['F21', 'F22', 'V21', 'V22', 'S21', 'S22'], 'make_log_prob_2G_B2x'),
            (['F31', 'F32', 'F33', 'V31', 'S31', 'S32', 'S33'], 'make_log_prob_3G_V32xV33xB3x'),
            (['F31', 'F32', 'F33', 'V31', 'S31', 'S32', 'S33', 'B3'], 'make_log_prob_3G_V32xV33x'),
        ]
        rng = np.random.default_rng(152)
        xx = np.linspace(-100, 100, 81)
        yy, errors = rng.normal(0, .01, len(xx)), np.linspace(.01, .02, len(xx))
        for names, factory in variants:
            with self.subTest(factory=factory):
                bounds = {n: ([0, 5] if n.startswith('F') else [-20, 20] if n.startswith('V')
                              else [-.1, .1] if n.startswith('B') else [2, 20] if n in ('S21', 'S32')
                              else [.5, 20] if n == 'S31' else [2, 100]) for n in names}
                model = Gmodel(xx, yy, errors, names, bounds)
                model.BB = .003
                lp, objective = getattr(model, factory)()
                u = rng.uniform(.02, .98, (20, len(names)))
                physical = map_params(u, model, 'u->x')
                np.testing.assert_allclose(map_params(physical, model, 'x->u'), u, atol=1e-13)
                broad, parent = ('S22', 'S21') if 'S21' in names else ('S33', 'S32')
                ib, ip = names.index(broad), names.index(parent)
                lower = physical[:, ip] + model.delta_v
                np.testing.assert_allclose(physical[:, ib], lower + u[:, ib] * (100 - lower))
                for endpoint in (0., 1.):
                    ue = u.copy()
                    ue[:, ib] = endpoint
                    xe = map_params(ue, model, 'u->x')
                    np.testing.assert_allclose(xe[:, ib], lower if endpoint == 0 else 100)
                for mapped, values in zip(u, physical):
                    p = dict(zip(names, values))
                    G = 2 if 'S21' in names else 3
                    predicted = np.full(len(xx), p.get(f'B{G}', model.BB))
                    for j in range(1, G + 1):
                        tag = f'{G}{j}'
                        center = p.get('V' + tag, p[f'V{G}1'])
                        width = p['S' + tag]
                        predicted += p['F' + tag] / (width*np.sqrt(2*np.pi)) * np.exp(-.5*((xx-center)/width)**2)
                    expected = -.5*np.sum(((yy-predicted)/errors)**2)
                    if G == 3:  # Existing narrow-width prior is intentionally preserved.
                        expected += np.log(p['S32']-model.delta_v-bounds['S31'][0])
                    self.assertAlmostEqual(lp(mapped), expected, places=8)
                    self.assertAlmostEqual(objective(mapped, yy), -expected, places=8)


if __name__ == '__main__':
    unittest.main()
