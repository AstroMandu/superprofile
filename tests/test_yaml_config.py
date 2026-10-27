"""YAML layering, typed errors, and reproducible file-relative paths."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import parse_args
from pipeline.yaml_config import load_config
from pipeline.models import GalaxyInput
from dataclasses import replace


class YamlConfigTest(unittest.TestCase):
    def test_named_output_is_under_ngfit_and_cli_can_override_name(self):
        self.path.write_text('output_dir_name: experiment\n')
        config = parse_args(['--config', str(self.path)])
        galaxy = GalaxyInput('TEST', self.path.parent / 'TEST/cube.fits', None, None, 'MOM')
        self.assertEqual(config.output_for(galaxy), galaxy.cube.parent / 'NGFIT/experiment')
        config = parse_args(['--config', str(self.path), '--output-dir-name', 'another'])
        self.assertEqual(config.output_for(galaxy), galaxy.cube.parent / 'NGFIT/another')
        explicit = replace(config, output_dir=self.path.parent / 'shared')
        self.assertEqual(explicit.output_for(galaxy), explicit.output_dir)
        for name in ('', '.', '..', '../outside', '/absolute', 'a/b', 'a\\b'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'output_dir_name'):
                replace(config, output_dir_name=name).validate()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'experiment.yaml'

    def test_paths_lists_and_cli_overrides(self):
        self.path.write_text('''data_dir: sample
output_dir: results
workers: 3
overwrite: true
fit:
  resamples: 100
masks:
  catalog: centers.csv
  regions: [angle]
  angles: [0, 90]
  angle_widths: [120]
''')
        config = parse_args(['--config', str(self.path), '--workers', '1', '--no-overwrite'])
        self.assertEqual(config.data_dir, self.path.parent / 'sample')
        self.assertEqual(config.output, self.path.parent / 'results')
        self.assertEqual(config.masks.catalog, self.path.parent / 'centers.csv')
        self.assertEqual(config.masks.angles, (0, 90))
        self.assertEqual(config.fit.resamples, 100)
        self.assertEqual(config.workers, 1)
        self.assertFalse(config.overwrite)
        self.assertEqual(config.masks.catalog, self.path.parent / 'centers.csv')

    def test_unknown_keys_and_wrong_types_fail_clearly(self):
        for contents, message in (
            ('workres: 2', 'unknown settings'),
            ('fit:\n  resampels: 100', 'unknown settings'),
            ('workers: "2"', 'expected int'),
            ('overwrite: "false"', 'expected bool'),
            ('masks:\n  angles: [true]', 'expected int'),
            ('masks: []', 'expected a YAML mapping'),
            ('[]', 'expected a YAML mapping'),
        ):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with self.assertRaisesRegex(ValueError, message):
                    load_config(self.path)

    def test_angle_range_and_explicit_cli_override(self):
        self.path.write_text('masks:\n  angles: {start: 0, stop: 360, step: 15}\n')
        config = parse_args(['--config', str(self.path)])
        self.assertEqual(config.masks.angles, tuple(range(0, 360, 15)))
        overridden = parse_args(['--config', str(self.path), '--angles', '30', '60'])
        self.assertEqual(overridden.masks.angles, (30, 60))
        for specification in ('{start: 0, stop: 360, step: 0}',
                              '{start: 0, stop: 360}',
                              '{start: 0, stop: 360, step: 1.5}',
                              '{start: 0, stop: 361, step: 15}',
                              '{start: 10, stop: 0, step: 15}'):
            with self.subTest(specification=specification):
                self.path.write_text(f'masks:\n  angles: {specification}\n')
                with self.assertRaises(ValueError):
                    load_config(self.path)

    def test_sample_change_clears_preset_galaxy_filter(self):
        self.path.write_text('preset: legacy-avid\ndata_dir: sample\n')
        self.assertIsNone(load_config(self.path).galaxies)

    def test_shipped_examples(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        angles = parse_args(['--config', str(root / 'viva_angles.yaml')])
        optical = parse_args(['--config', str(root / 'viva_optical.yaml')])
        self.assertEqual(len(angles.masks.angles), 24)
        self.assertEqual(angles.masks.regions, ('angle',))
        self.assertEqual(optical.masks.regions, ('inner', 'outer'))
        self.assertTrue(optical.masks.is_optical_catalog)


if __name__ == '__main__':
    unittest.main()
