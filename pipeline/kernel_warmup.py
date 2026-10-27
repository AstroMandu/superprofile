"""Prepare the on-disk likelihood cache in a short-lived process."""
import os
from pathlib import Path
import subprocess
import sys


def prepare_kernel_cache():
    # Keep numerical imports/compiler memory out of the orchestration process.
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    subprocess.run([sys.executable, '-m', 'pipeline.kernel_warmup'],
                   env=env, check=True)


def main():
    from .runtime import configure_runtime
    configure_runtime()
    from xgfit.gmodel.kernels import warmup_likelihood_kernel
    warmup_likelihood_kernel()
    print('[numba] Shared likelihood cache ready', flush=True)


if __name__ == '__main__':
    main()
