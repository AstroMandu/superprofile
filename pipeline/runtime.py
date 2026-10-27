"""Process environment; call before importing numerical libraries."""
import os


def configure_runtime():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    # Do not spawn nested joblib workers inside fit workers.
    os.environ.setdefault('JOBLIB_MULTIPROCESSING', '0')
    os.environ.setdefault('ASTROPY_SKIP_CONFIG_UPDATE', '1')
    os.environ.setdefault('MPLBACKEND', 'Agg')
