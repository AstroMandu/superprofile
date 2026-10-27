"""Local serial/process execution and optional BAYGAUD classification."""
import multiprocessing as mp
import os
from collections import deque
from queue import Queue

from tqdm import tqdm


def run_tasks(function, tasks, workers, description, show_progress=True):
    if not tasks:
        return
    if workers == 1 or len(tasks) == 1:
        for task in tqdm(tasks, desc=description, disable=not show_progress):
            yield function(task)
    else:
        context = mp.get_context('forkserver')
        with context.Pool(processes=min(workers, len(tasks)), maxtasksperchild=1) as pool:
            yield from tqdm(pool.imap_unordered(function, tasks), total=len(tasks), desc=description, disable=not show_progress)


def run_fit_tasks(function, tasks, workers, show_progress=True):
    """Release a galaxy's dependents only after the caller commits its whole fit.

    The caller must save and validate each yielded result before advancing this
    iterator. Keep only worker-count tasks submitted so released rings can start
    without waiting behind every other queued whole fit.
    """
    whole_names = {task.stack.galaxy.name for task in tasks if task.stack.region == 'whole'}
    ready, blocked = deque(), {}
    for task in tasks:
        name = task.stack.galaxy.name
        if task.stack.region != 'whole' and task.settings.guess_from_whole and name in whole_names:
            blocked.setdefault(name, []).append(task)
        else:
            ready.append(task)

    def release(task):
        if task.stack.region == 'whole':
            ready.extendleft(reversed(blocked.pop(task.stack.galaxy.name, [])))

    with tqdm(total=len(tasks), desc='stack + fit', disable=not show_progress) as bar:
        if workers == 1 or len(tasks) <= 1:
            while ready:
                task = ready.popleft()
                yield function(task)
                release(task)
                bar.update(1)
            return
        results = Queue()
        context = mp.get_context('forkserver')
        with context.Pool(processes=min(workers, len(tasks)), maxtasksperchild=1) as pool:
            active = 0
            while ready or active:
                while ready and active < workers:
                    task = ready.popleft()
                    pool.apply_async(
                        function, (task,),
                        callback=lambda key, task=task: results.put((task, key, None)),
                        error_callback=lambda error, task=task: results.put((task, None, error)),
                    )
                    active += 1
                task, key, error = results.get()
                active -= 1
                if error is not None:
                    raise error
                yield key
                release(task)
                bar.update(1)


def classify_galaxy(task):
    galaxy, settings = task
    if galaxy.classified is None:
        return galaxy.name
    if galaxy.classified.exists() and not settings.overwrite_classify:
        return galaxy.name
    from class_baypi_classify import Classify
    classifier = Classify(galaxy.cube, path_baygaud=os.getenv('BAY'),
                          vdisp_low_intrinsic=settings.vdisp_low_intrinsic)
    previous_directory = os.getcwd()
    try:
        classifier.run(snlim_peak=settings.snlim_peak, key_classify=settings.key_classify,
                       remove_unnecessary=True)
    finally:
        # The legacy classifier changes cwd; serial execution must not leak it.
        os.chdir(previous_directory)
    if not galaxy.classified.is_dir():
        raise FileNotFoundError(f'Classification did not create {galaxy.classified}')
    return galaxy.name
