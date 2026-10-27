"""Behavioral checks for planning, dependencies, and durable resume semantics."""
import sys
import io
from threading import Barrier
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import FitConfig, MaskConfig, PipelineConfig, parse_args
from pipeline.models import GalaxyInput
from pipeline.planning import discover_galaxies, plan_jobs
from pipeline.results import ProgressMonitor, ResultStore, completed_keys
from pipeline.runner import check_whole_prerequisites, prepare_masks, run_pipeline


class PipelineTest(unittest.TestCase):
    def test_default_outputs_are_per_cube_and_resume_independently(self):
        second = self.config.data_dir / 'GAL2'
        second.mkdir()
        (second / 'cube.fits').touch()
        (second / 'hermite.npy').touch()
        config = replace(self.config, output_dir=None, workers=1,
                         fit=replace(self.config.fit, guess_from_whole=True),
                         masks=replace(self.config.masks, regions=('whole', 'inner')))
        seen = []

        def fake_fit(task):
            name = task.stack.galaxy.name
            self.assertEqual(task.output, task.stack.galaxy.cube.parent / 'NGFIT')
            self.assertEqual(task.temp.parent, task.output)
            if task.stack.region != 'whole':
                self.assertIn((name, '_'), completed_keys(task.output))
            seen.append(task.stack.key)
            frame = pd.DataFrame({'Name': [name], 'suffix': [task.stack.suffix], 'S1': [10.]})
            for kind in ('gfit', 'para'):
                frame.to_string(task.temp / f'{name}{task.stack.suffix}_plotfitter_temp_{kind}.csv', index=False)
            return task.stack.key

        with patch('pipeline.runner.validate_catalogs', side_effect=lambda config, galaxies: galaxies), patch('pipeline.runner.prepare_mask'), \
             patch('pipeline.runner.write_geometry'), patch('pipeline.worker.fit_stack', side_effect=fake_fit):
            run_pipeline(config)
            self.assertEqual(len(seen), 4)
            for name in ('GAL1', 'GAL2'):
                output = config.data_dir / name / 'NGFIT'
                self.assertEqual(completed_keys(output), {(name, '_'), (name, '_I1.0Ropt')})
                status = pd.read_csv(output / 'Plotfit_stat.csv', sep=r'\s+', comment='#')
                self.assertEqual(set(status['Name']), {name})
                self.assertEqual(set(status['Status']), {'Done'})
                self.assertTrue((output / 'last_run.json').exists())
                self.assertEqual(list(output.glob('temp_*')), [])
            run_pipeline(config)
            self.assertEqual(len(seen), 4)

    def test_masks_parallelize_galaxies_with_separate_caches(self):
        galaxies = [self.galaxies[0], replace(self.galaxies[0], name='GAL2',
                                            cube=self.root / 'GAL2/cube.fits')]
        jobs = plan_jobs(self.config, galaxies)
        barrier = Barrier(2)
        caches = {}
        seen = []

        def prepare(job, config, cache=None):
            name = job.galaxy.name
            if name not in caches:
                caches[name] = cache
                barrier.wait(timeout=5)
            self.assertIs(cache, caches[name])
            seen.append(job.key)

        with patch('pipeline.runner.prepare_mask', side_effect=prepare):
            prepare_masks(jobs, self.config.masks, workers=2)
        self.assertCountEqual(seen, [job.key for job in jobs])
        self.assertIsNot(caches['GAL1'], caches['GAL2'])

    def test_terminal_progress_reports_changes_without_status_csv(self):
        config = replace(self.config, progress='terminal')
        temp = self.root / 'status'
        temp.mkdir()
        jobs = plan_jobs(config, self.galaxies)
        monitor = ProgressMonitor(config, temp, jobs)
        name, suffix = jobs[0].key
        (temp / 'stat_test.txt').write_text(f'{name} {suffix} Resampling 12/100 00:30')
        with patch('pipeline.results.tqdm.write') as write:
            monitor.snapshot()
            self.assertIn('12/100', write.call_args.args[0])
            self.assertIn('00:30', write.call_args.args[0])
            monitor.snapshot()
            self.assertEqual(write.call_count, 1)
            monitor.mark_completed(jobs[0].key)
            monitor.snapshot()
            self.assertIn('Done', write.call_args.args[0])
            self.assertEqual(write.call_count, 2)
        self.assertFalse((config.output / 'Plotfit_stat.csv').exists())
        stream = io.StringIO()
        with patch('pipeline.results.sys.stdout', stream), \
             patch.object(stream, 'isatty', return_value=True):
            live = ProgressMonitor(config, temp, jobs)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with live:
                    live.mark_completed(jobs[0].key)
                    live.snapshot()
                    raise RuntimeError('interrupted')
        rendered = stream.getvalue()
        self.assertIn('\033[?1049h', rendered)
        self.assertGreaterEqual(rendered.count('\033[H\033[J'), 2)
        self.assertIn('Done', rendered)
        self.assertTrue(rendered.endswith('\033[?25h\033[?1049l'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder = self.root / 'input' / 'GAL1'
        folder.mkdir(parents=True)
        (folder / 'cube.fits').touch()
        (folder / 'hermite.npy').touch()
        self.config = PipelineConfig(data_dir=folder.parent, output_dir=self.root / 'output')
        self.galaxies = discover_galaxies(self.config)

    def test_optical_jobs_and_multiple_radii(self):
        config = replace(self.config, masks=replace(self.config.masks, multipliers=(0.5, 1.0)))
        jobs = plan_jobs(config, self.galaxies)
        self.assertEqual({job.suffix for job in jobs},
                         {'_I0.5Ropt', '_O0.5Ropt', '_I1.0Ropt', '_O1.0Ropt'})
        for job in jobs:
            self.assertEqual(job.mask.name, f'mask_IO{float(job.multiplier)}Ropt.fits')
        self.assertEqual(len({job.mask for job in jobs}), 2)

    def test_legacy_regions_are_independent(self):
        masks = MaskConfig(catalog=Path('/home/mskim/workspace/research/data/catalog/cat_diameters.csv'),
                   regions=('whole', 'inner', 'outer', 'ring', 'angle'),
                           multipliers=(0.6,), angles=(0, 90), angle_widths=(120,))
        jobs = plan_jobs(replace(self.config, masks=masks), self.galaxies)
        self.assertEqual({job.suffix for job in jobs},
                         {'_', '_I0.6RHI', '_O0.6RHI', '_R0.60RHI', '_W120_A000', '_W120_A090'})
        ring_only = replace(self.config, masks=replace(masks, regions=('ring',)))
        self.assertEqual(len(plan_jobs(ring_only, self.galaxies)), 1)

    def test_whole_fit_dependency(self):
        config = replace(self.config, fit=replace(self.config.fit, guess_from_whole=True))
        jobs = plan_jobs(config, self.galaxies)
        with self.assertRaises(ValueError):
            check_whole_prerequisites(config, jobs, set())
        check_whole_prerequisites(config, jobs, {('GAL1', '_')})
        whole_config = replace(config, masks=replace(config.masks, regions=('whole', 'inner')))
        check_whole_prerequisites(whole_config, plan_jobs(whole_config, self.galaxies), set())

    def test_missing_velocity_field_skips_only_that_galaxy(self):
        self.galaxies[0].secondary_vf.unlink()
        second = self.config.data_dir / 'GAL2'
        second.mkdir()
        (second / 'cube.fits').touch()
        (second / 'hermite.npy').touch()
        self.assertEqual([galaxy.name for galaxy in discover_galaxies(self.config)], ['GAL2'])
        self.assertFalse(self.config.output.exists())

    def test_missing_explicit_cube_is_skipped(self):
        config = replace(self.config, galaxies=('MISSING',))
        self.assertEqual(discover_galaxies(config), [])
        self.assertFalse(config.output.exists())

    def test_dry_run_has_no_writes(self):
        config = replace(self.config, dry_run=True, masks=replace(self.config.masks, regions=('whole',)))
        jobs = run_pipeline(config)
        self.assertEqual(len(jobs), 1)
        self.assertFalse(config.output.exists())
        self.assertFalse((self.galaxies[0].cube.parent / 'NGFIT' / 'mask').exists())

    def test_resume_requires_both_tables_and_merges_idempotently(self):
        output = self.config.output
        output.mkdir()
        temp = output / 'temp_test'
        temp.mkdir()
        store = ResultStore(output, temp)
        frame = pd.DataFrame({'Name': ['GAL1'], 'suffix': ['_I1.0Ropt'], 'S1': [10.]})
        frame.to_string(temp / 'GAL1_I1.0Ropt_plotfitter_temp_gfit.csv', index=False)
        store.merge_pending()
        self.assertEqual(completed_keys(output), set())
        for kind in ('gfit', 'para'):
            frame.to_string(temp / f'GAL1_I1.0Ropt_plotfitter_temp_{kind}.csv', index=False)
        # A failed second-table commit must preserve both fragments for replay.
        import pipeline.results as results
        actual_write = results.atomic_table
        def fail_second(frame, path):
            if path.name.endswith('_params.csv'):
                raise OSError('simulated disk failure')
            actual_write(frame, path)
        with patch('pipeline.results.atomic_table', side_effect=fail_second):
            with self.assertRaises(OSError):
                store.merge_pending()
        self.assertEqual(len(list(temp.glob('*.csv'))), 2)
        store.merge_pending()
        self.assertEqual(completed_keys(output), {('GAL1', '_I1.0Ropt')})
        self.assertEqual(len(pd.read_csv(output / 'info_stacked_GFIT.csv', sep=r'\s+')), 1)
        self.assertEqual(list(temp.glob('*.csv')), [])

    def test_runner_commits_whole_before_regions_and_resumes(self):
        config = replace(self.config, workers=1,
                         fit=replace(self.config.fit, guess_from_whole=True),
                         masks=replace(self.config.masks, regions=('inner', 'whole')))
        seen = []
        def fake_fit(task):
            if task.stack.region == 'inner':
                self.assertIn(('GAL1', '_'), completed_keys(task.output))
            seen.append(task.stack.region)
            frame = pd.DataFrame({'Name': ['GAL1'], 'suffix': [task.stack.suffix], 'S1': [10.]})
            for kind in ('gfit', 'para'):
                frame.to_string(task.temp / f'GAL1{task.stack.suffix}_plotfitter_temp_{kind}.csv', index=False)
            return task.stack.key
        with patch('pipeline.runner.validate_catalogs', side_effect=lambda config, galaxies: galaxies), patch('pipeline.runner.prepare_mask'), \
             patch('pipeline.runner.write_geometry'), patch('pipeline.worker.fit_stack', side_effect=fake_fit):
            run_pipeline(config)
            self.assertEqual(seen, ['whole', 'inner'])
            self.assertTrue((config.output / 'last_run.json').is_file())
            status = pd.read_csv(config.output / 'Plotfit_stat.csv', sep=r'\s+', comment='#')
            self.assertEqual(len(status), 2)
            self.assertEqual(set(status['Status']), {'Done'})
            self.assertEqual(list(config.output.glob('temp_*')), [])
            run_pipeline(config)
            self.assertEqual(seen, ['whole', 'inner'])

    def test_runner_failure_preserves_completed_results_and_run_manifest(self):
        config = replace(self.config, workers=1,
                         masks=replace(self.config.masks, regions=('whole', 'inner')))
        def fake_fit(task):
            if task.stack.region == 'inner':
                raise RuntimeError('simulated fit failure')
            frame = pd.DataFrame({'Name': ['GAL1'], 'suffix': ['_'], 'S1': [10.]})
            for kind in ('gfit', 'para'):
                frame.to_string(task.temp / f'GAL1_plotfitter_temp_{kind}.csv', index=False)
            return task.stack.key
        with patch('pipeline.runner.validate_catalogs', side_effect=lambda config, galaxies: galaxies), patch('pipeline.runner.prepare_mask'), \
             patch('pipeline.runner.write_geometry'), patch('pipeline.worker.fit_stack', side_effect=fake_fit):
            with self.assertRaisesRegex(RuntimeError, 'simulated fit failure'):
                run_pipeline(config)
        self.assertEqual(completed_keys(config.output), {('GAL1', '_')})
        temporary = list(config.output.glob('temp_*'))
        self.assertEqual(len(temporary), 1)
        self.assertTrue((temporary[0] / 'run.json').is_file())

    def test_cli_preserves_optical_command_and_allows_explicit_legacy(self):
        optical = parse_args(['--data-dir', str(self.config.data_dir), '--workers', '1'])
        self.assertEqual(optical.data_dir, self.config.data_dir)
        self.assertEqual(optical.masks.regions, ('inner', 'outer'))
        legacy = parse_args(['--radius-column', 'r25', '--regions', 'ring', '--multipliers', '0.5'])
        self.assertEqual(legacy.masks.catalog_kind, 'legacy')
        self.assertEqual(legacy.masks.radius_column, 'r25')
        self.assertFalse(legacy.fit.guess_from_whole)


if __name__ == '__main__':
    unittest.main()
