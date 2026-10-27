"""Execute one stack and fit; all worker state arrives in a FitJob."""
import gc
import logging
import os

from .models import FitJob


def close_stack_logger(stack):
    """Release file handlers and the logging registry's strong reference."""
    logger = getattr(stack, 'logger', None)
    if logger is None:
        return
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if logging.Logger.manager.loggerDict.get(logger.name) is logger:
        logging.Logger.manager.loggerDict.pop(logger.name, None)
    stack.logger = None


def fit_stack(task: FitJob):
    # Lazy numerical imports keep planning, CLI help, and worker spawning light.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shiftnstack import ShiftnStack
    from xgfit.xgfit import XGFIT
    from .galaxy_settings import dict_config_stack
    from .initialization import apply_whole_guess

    job, settings = task.stack, task.settings
    galaxy = job.galaxy
    if not galaxy.cube.is_file():
        raise FileNotFoundError(galaxy.cube)
    if galaxy.classified is not None and not galaxy.classified.is_dir():
        raise FileNotFoundError(galaxy.classified)
    if galaxy.secondary_vf is not None and not galaxy.secondary_vf.is_file():
        raise FileNotFoundError(galaxy.secondary_vf)
    if job.mask is not None and not job.mask.is_file():
        raise FileNotFoundError(job.mask)
    correct_vf = dict_config_stack.get(galaxy.name, {}).get('correct_secondary_vf', False)
    stack = fitter = None
    try:
        stack = ShiftnStack(
            galaxy.cube, galaxy.classified, path_mask=job.mask,
            mask_value=job.mask_value,
            path_vf_secondary=galaxy.secondary_vf, correct_vf_secondary=correct_vf,
            secondary_mode=galaxy.secondary_mode,
        )
        stack.path_temp = task.temp
        stack.suffix = job.suffix
        # Spectral exceptions remain disabled, as in the original pipeline.
        stack.setup_logger(suffix=job.suffix)
        stack.run(stack_secondary=True)
        stack.log_summary()
        stack.df_stacked.to_string(task.output / f'{galaxy.name}{job.suffix}_stacked.csv', index=False)
        fitter = XGFIT(
            stack.df_stacked, dict_stacked=stack.dict_stacked, name_cube=galaxy.name,
            path_plot=task.output / 'png', path_temp=task.temp,
            plot_autocorr=False, vdisp_low_intrinsic=settings.vdisp_low_intrinsic,
        )
        close_stack_logger(stack)
        stack = None
        gc.collect()
        fitter.initmod = settings.initmod
        fitter.truth_from_resampling = settings.truth_from_resampling
        fitter.fallback_to_2gfit = settings.fallback_to_2gfit
        fitter.method_minimize = settings.method
        fitter.statistics = settings.statistics
        fitter.intermediate_plots = settings.intermediate_plots
        fitter.correlation_diagnostics = settings.correlation_diagnostics
        fitter.walk_plot_max_steps = settings.walk_plot_max_steps
        fitter.walk_plot_max_walkers = settings.walk_plot_max_walkers
        apply_whole_guess(fitter, task)
        fitter.run(suffix=job.suffix, pbar_resample=task.show_progress,
                   nsample_resample=settings.resamples, NGFIT=settings.model)
        for frame, kind in ((fitter.df.copy(), 'gfit'), (fitter.df_params.copy(), 'para')):
            frame['suffix'] = job.suffix
            columns = frame.columns.tolist()
            columns.insert(1, columns.pop(columns.index('suffix')))
            target = task.temp / f'{galaxy.name}{job.suffix}_plotfitter_temp_{kind}.csv'
            pending = target.with_suffix('.pending')
            frame[columns].to_string(pending, index=False)
            os.replace(pending, target)
        return job.key
    finally:
        try:
            if stack is not None:
                close_stack_logger(stack)
            if fitter is not None:
                fitter.cleanup()
        except Exception as error:
            print(f'[cleanup] {galaxy.name}{job.suffix}: {error}', flush=True)
        finally:
            plt.close('all')
            gc.collect()
