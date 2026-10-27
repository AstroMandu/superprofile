"""Experiment lifecycle: validate → plan → resume → masks → fits → results."""
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PipelineConfig
from .execution import classify_galaxy, run_tasks, run_fit_tasks
from .masks import prepare_mask, write_geometry
from .models import FitJob
from .planning import discover_galaxies, plan_jobs, validate_catalogs
from .results import ProgressMonitor, ResultStore, completed_keys, write_manifest


@contextmanager
def report_stage(label):
    """Identify slow operations, including time spent inside a blocking call."""
    started = time.monotonic()
    stop = Event()
    print(f'[start] {label}', flush=True)

    def heartbeat():
        while not stop.wait(15):
            print(f'[waiting {time.monotonic() - started:.0f}s] {label}', flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        print(f'[failed after {time.monotonic() - started:.1f}s] {label}', flush=True)
        raise
    else:
        print(f'[done {time.monotonic() - started:.1f}s] {label}', flush=True)
    finally:
        stop.set()
        thread.join()


def check_whole_prerequisites(config, jobs, completed):
    if not config.fit.guess_from_whole:
        return
    planned_whole = {job.galaxy.name for job in jobs if job.region == 'whole'}
    missing = {job.galaxy.name for job in jobs if job.region != 'whole'
               and (job.galaxy.name, '_') not in completed and job.galaxy.name not in planned_whole}
    if missing:
        raise ValueError('Whole-galaxy initialization requires --regions whole ... or existing whole fits: '
                         + ', '.join(sorted(missing)))


def prepare_masks(jobs, config, workers):
    """Own each galaxy's files and cache in one task to avoid write races."""
    groups = {}
    for job in jobs:
        if job.region != 'whole':
            groups.setdefault(job.galaxy.cube, []).append(job)
    # Legacy helpers have not been audited for thread safety.
    count = min(workers, len(groups)) if config.is_optical_catalog else min(1, len(groups))
    print(f'[masks] {sum(map(len, groups.values()))} selections, '
          f'{len(groups)} galaxies; {count} preparation workers', flush=True)

    def prepare_galaxy(batch):
        cache = {}  # Released after this galaxy; never reused across runs.
        maskdir = batch[0].galaxy.cube.parent / 'NGFIT' / 'mask'
        before = {path: path.stat() for path in maskdir.glob('mask*.fits')}
        with report_stage(f'Masks for {batch[0].galaxy.name}: {len(batch)} selections'):
            for index, job in enumerate(batch, 1):
                started = time.monotonic()
                print(f'[mask start {index}/{len(batch)}] {job.galaxy.name}{job.suffix}', flush=True)
                prepare_mask(job, config, cache=cache)
                print(f'[mask done {time.monotonic() - started:.2f}s] '
                      f'{job.galaxy.name}{job.suffix}', flush=True)

        # Include companion/radial masks written by the generators, while
        # leaving unrelated masks from earlier experiments alone.
        changed = {path for path in maskdir.glob('mask*.fits')
                   if path not in before or path.stat() != before[path]}
        return changed | {job.mask for job in batch if job.mask is not None}

    used_masks = set()
    if count <= 1:
        for batch in groups.values():
            used_masks.update(prepare_galaxy(batch))
    else:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix='mask') as pool:
            futures = [pool.submit(prepare_galaxy, batch) for batch in groups.values()]
            for future in as_completed(futures):
                used_masks.update(future.result())
    return used_masks


def cleanup_masks(paths):
    """Delete explicit masks owned by a successful run, never whole folders."""
    count = 0
    for path in sorted(set(paths)):
        if path.is_file() or path.is_symlink():
            path.unlink()
            count += 1
    print(f'[masks] Removed {count} mask files after successful completion', flush=True)


def describe_plan(config, jobs, completed):
    print(f'Input:  {config.data_dir}')
    layout = '<each cube directory>/NGFIT'
    if config.output_dir_name is not None:
        layout += f'/{config.output_dir_name}'
    print(f'Output: {config.output_dir or layout}')
    print(f'Fit: {config.fit.mode} / {config.fit.model}; up to {config.fit.resamples} resamples')
    for job in jobs:
        state = 'skip (complete)' if job.key in completed and not config.overwrite else 'run'
        print(f'  {job.galaxy.name:16} {job.suffix:24} {state}')


def run_pipeline(config: PipelineConfig):
    started = time.monotonic()
    with report_stage('Validate configuration and discover galaxies'):
        config.validate()
        galaxies = discover_galaxies(config)
    with report_stage(f'Validate catalogs for {len(galaxies)} galaxies'):
        galaxies = validate_catalogs(config, galaxies)
    if not galaxies:
        print('No galaxies with catalog entries; nothing to run.', flush=True)
        return []
    with report_stage('Plan selections and read existing results'):
        ring_records = {}
        planned = plan_jobs(config, galaxies, ring_records=ring_records)
        outputs = {galaxy.name: config.output_for(galaxy) for galaxy in galaxies}
        complete = {key for name, output in outputs.items()
                    for key in completed_keys(output) if key[0] == name}
    pending = [job for job in planned if config.overwrite or job.key not in complete]
    check_whole_prerequisites(config, pending, complete)
    describe_plan(config, planned, complete)
    if config.dry_run:
        return planned
    if ring_records:
        import pandas as pd
        for output in set(outputs.values()):
            records = [row for name, rows in ring_records.items() if outputs[name] == output for row in rows]
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output / 'ring_geometry.csv', index=False)
    if not pending:
        print('All selections are complete; nothing to run.')
        return []

    stores = {}
    for output in {outputs[job.galaxy.name] for job in pending}:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'png').mkdir(exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix='temp_', dir=output))
        stores[output] = ResultStore(output, temp)
        local_jobs = [job for job in planned if outputs[job.galaxy.name] == output]
        write_manifest(replace(config, output_dir=output), local_jobs, temp)
    destinations = {name: (output, stores[output].temp) for name, output in outputs.items()
                    if output in stores}
    try:
        if config.fit.classify:
            needed_names = {job.galaxy.name for job in pending}
            classify_tasks = [(galaxy, config.fit) for galaxy in galaxies
                              if galaxy.name in needed_names and galaxy.classified is not None]
            with report_stage(f'Classification: {len(classify_tasks)} galaxies, up to {min(config.workers, len(classify_tasks))} worker processes'):
                for name in run_tasks(classify_galaxy, classify_tasks, config.workers, 'classify'):
                    print(f'[classify] {name}: ready', flush=True)
        if set(config.masks.regions) != {'whole'}:
            with report_stage('Write optical geometry'):
                for output in stores:
                    local_galaxies = [galaxy for galaxy in galaxies if outputs[galaxy.name] == output]
                    write_geometry(config.masks, local_galaxies, output)
        used_masks = prepare_masks(pending, config.masks, config.workers)
        with report_stage('Import stacking and fitting libraries'):
            from .worker import fit_stack
        if min(config.workers, len(pending)) > 1:
            from .kernel_warmup import prepare_kernel_cache
            with report_stage('Prepare shared Numba likelihood cache'):
                prepare_kernel_cache()
        with ProgressMonitor(config, temp, pending, destinations=destinations) as progress:
            progress.phase = (f'{len(pending)} selections; up to {min(config.workers, len(pending))} workers; '
                              'regional fits released after their galaxy whole fit is saved')
            print(f'[fits] {progress.phase}', flush=True)
            tasks = [FitJob(job, config.fit, *destinations[job.galaxy.name],
                            len(pending) == 1 and config.progress != 'terminal') for job in pending]
            for key in run_fit_tasks(fit_stack, tasks, config.workers,
                                     show_progress=config.progress != 'terminal'):
                output = outputs[key[0]]
                stores[output].merge_pending()
                if key not in completed_keys(output):
                    raise RuntimeError(f'Worker finished without both result rows: {key}')
                progress.mark_completed(key)
        for store in stores.values():
            store.merge_pending()
        complete = set().union(*(completed_keys(output) for output in stores))
        missing = {job.key for job in pending} - complete
        if missing:
            raise RuntimeError(f'Incomplete fits: {sorted(missing)}')
        # Preserve the exact experiment configuration after successful cleanup.
        for output, store in stores.items():
            (output / 'last_run.json').write_text((store.temp / 'run.json').read_text())
    except BaseException:
        for store in stores.values():
            store.merge_pending()
            print(f'Run failed; intermediate files preserved in {store.temp}', flush=True)
        raise
    else:
        for store in stores.values():
            shutil.rmtree(store.temp)
        if config.masks.cleanup_after_run:
            cleanup_masks(used_masks)
    print(f'Finished {len(pending)} selections in {timedelta(seconds=time.monotonic() - started)}')
    return pending
