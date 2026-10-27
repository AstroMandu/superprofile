"""Durable result tables, resumability, and progress snapshots."""
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock, Thread

import pandas as pd
from natsort import natsort_keygen
from tqdm import tqdm

TABLES = {
    'gfit': 'info_stacked_GFIT.csv',
    'para': 'info_stacked_GFIT_params.csv',
}


def atomic_table(frame, path):
    pending = path.with_suffix(path.suffix + '.pending')
    frame.to_string(pending, index=False)
    os.replace(pending, path)


def completed_keys(output):
    """A selection is complete only when both result tables contain its key."""
    keys = []
    for filename in TABLES.values():
        path = output / filename
        if not path.exists():
            return set()
        frame = pd.read_csv(path, sep=r'\s+')
        keys.append(set(zip(frame['Name'], frame['suffix'])))
    return set.intersection(*keys)


class ResultStore:
    def __init__(self, output, temp):
        self.output = output
        self.temp = temp

    def merge_pending(self):
        """Commit tables before deleting worker fragments; safe to replay after failure."""
        consumed = []
        for kind, filename in TABLES.items():
            fragments = sorted(self.temp.glob(f'*_plotfitter_temp_{kind}.csv'))
            if not fragments:
                continue
            target = self.output / filename
            frames = [pd.read_csv(target, sep=r'\s+')] if target.exists() else []
            frames.extend(pd.read_csv(path, sep=r'\s+') for path in fragments)
            frame = pd.concat(frames, ignore_index=True)
            frame.drop_duplicates(subset=['Name', 'suffix'], keep='last', inplace=True)
            frame.sort_values(['Name', 'suffix'], key=natsort_keygen(), inplace=True)
            atomic_table(frame, target)
            consumed.extend(fragments)
        # If either table fails, all fragments remain for recovery.
        for path in consumed:
            path.unlink()


def write_manifest(config, jobs, temp):
    manifest = {
        'config': asdict(config),
        'output': str(config.output),
        'jobs': [asdict(job) for job in jobs],
    }
    (temp / 'run.json').write_text(json.dumps(manifest, default=str, indent=2) + '\n')


class ProgressMonitor:
    """Read fitter-owned status files; result merging stays in the parent process."""
    def __init__(self, config, temp, jobs=(), destinations=None):
        self.lock = Lock()
        self.finished = set()
        self.displayed = {}
        self.phase = 'Starting fits'
        self.live = config.progress == 'terminal' and sys.stdout.isatty()
        self.rows = {job.key: [*job.key, "Pending", "-", "-"] for job in jobs}
        self.config = config
        self.temp = temp
        self.destinations = destinations
        self.stop = Event()
        self.thread = Thread(target=self._run, name='superprofile-progress', daemon=True)

    def mark_completed(self, key):
        with self.lock:
            self.finished.add(key)
            self.rows[key] = [*key, "Done", "-", "-"]

    def snapshot(self):
        updates = {}
        temps = {pair[1] for pair in self.destinations.values()} if self.destinations else {self.temp}
        for path in sorted(path for temp in temps for path in temp.glob('stat*.txt')):
            try:
                fields = path.read_text().split()
            except FileNotFoundError:
                continue
            if len(fields) >= 5:
                updates[tuple(fields[:2])] = fields[:5]
        with self.lock:
            self.rows.update({key: row for key, row in updates.items() if key not in self.finished})
            rows = list(self.rows.values())
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=['Name', 'Suffix', 'Status', 'Iteration', 'ETA'])
        frame.sort_values(['Name', 'Suffix'], key=natsort_keygen(), inplace=True)
        if self.config.progress == 'terminal':
            if self.live:
                done = sum(row[2] == 'Done' for row in rows)
                size = shutil.get_terminal_size()
                # Keep active fits visible even when the queue exceeds screen height.
                order = frame['Status'].map(lambda status: 2 if status == 'Done' else 1 if status == 'Pending' else 0)
                visible = frame.iloc[order.argsort(kind='stable')].head(max(1, size.lines - 6))
                text = (f'[stack + fit: {done}/{len(rows)} complete]\n'
                        + self.phase + '\n'
                        + visible.to_string(index=False))
                if len(visible) < len(frame):
                    text += f'\nShowing {len(visible)}/{len(frame)} selections (active first)'
                text = '\n'.join(line[:max(1, size.columns - 1)] for line in text.splitlines())
                sys.stdout.write('\033[H\033[J' + text + '\n')
                sys.stdout.flush()
                return
            changed = []
            for row in frame.itertuples(index=False, name=None):
                key = row[:2]
                if self.displayed.get(key) != row:
                    changed.append(row)
                    self.displayed[key] = row
            if changed:
                done = sum(row[2] == 'Done' for row in rows)
                table = pd.DataFrame(changed, columns=frame.columns)
                tqdm.write(f'[stack + fit: {done}/{len(rows)} complete]\n'
                           + table.to_string(index=False))
            return
        outputs = ({pair[0] for pair in self.destinations.values()}
                   if self.destinations else {self.config.output})
        for output in outputs:
            selected = frame
            if self.destinations:
                names = [name for name, pair in self.destinations.items() if pair[0] == output]
                selected = frame.loc[frame['Name'].isin(names)]
            self.write_status(selected, output)

    def write_status(self, frame, output):
        target = output / 'Plotfit_stat.csv'
        pending = target.with_suffix('.pending')
        with pending.open('w') as stream:
            stream.write(f'# {output.name}\n')
            stream.write(f'# workers = {self.config.workers}\n')
            stream.write(f'# max_resamples = {self.config.fit.resamples}\n')
            stream.write(f'# mode = {self.config.fit.mode}\n')
            frame.to_string(stream, index=False)
        os.replace(pending, target)

    def _run(self):
        while not self.stop.wait(1):
            self.snapshot()

    def __enter__(self):
        if self.live:
            sys.stdout.write('\033[?1049h\033[?25l')
            sys.stdout.flush()
        self.snapshot()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        try:
            self.snapshot()
        finally:
            if self.live:
                sys.stdout.write('\033[?25h\033[?1049l')
                sys.stdout.flush()
