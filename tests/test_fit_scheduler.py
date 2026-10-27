"""Per-galaxy dependencies release after durable parent-side commits."""
import sys
import unittest
from pathlib import Path
from multiprocessing.pool import ThreadPool
from threading import Event
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import FitConfig
from pipeline.models import GalaxyInput, StackJob, FitJob
from pipeline.execution import run_fit_tasks


class SchedulerTest(unittest.TestCase):
    def tasks(self):
        tasks = []
        for name in ('A', 'B'):
            galaxy = GalaxyInput(name, Path(f'/tmp/{name}/cube.fits'), None, None, 'MOM')
            for region, suffix in (('whole', '_'), ('ring', '_R0-1beam')):
                tasks.append(FitJob(StackJob(galaxy, region, suffix),
                                    FitConfig(guess_from_whole=True), Path('/tmp'), Path('/tmp')))
        return tasks

    def test_rings_start_while_another_whole_is_running_after_commit(self):
        ring_started, committed = Event(), Event()

        def fit(task):
            name, region = task.stack.galaxy.name, task.stack.region
            if name == 'B' and region == 'whole':
                if not ring_started.wait(5):
                    raise RuntimeError('A ring was blocked behind B whole')
            if name == 'A' and region == 'ring':
                self.assertTrue(committed.is_set())
                ring_started.set()
            return task.stack.key

        with patch('pipeline.execution.mp.get_context') as context:
            context.return_value.Pool.side_effect = lambda processes, **kwargs: ThreadPool(processes)
            seen = []
            for key in run_fit_tasks(fit, self.tasks(), workers=2, show_progress=False):
                seen.append(key)
                if key == ('A', '_'):
                    committed.set()
        self.assertCountEqual(seen, [task.stack.key for task in self.tasks()])

    def test_failed_whole_never_releases_its_rings(self):
        seen = []

        def fail(task):
            seen.append(task.stack.region)
            raise RuntimeError('whole failed')

        with self.assertRaisesRegex(RuntimeError, 'whole failed'):
            list(run_fit_tasks(fail, self.tasks(), workers=1, show_progress=False))
        self.assertEqual(seen, ['whole'])
