# Expects on self: path_plot, name_cube, suffix, list_disp, list_NHI_,
#                  df, df_params, gmodel, gmodel_1G, sampler, resampled,
#                  burnin, thin
from __future__ import annotations

import gc

import matplotlib.pyplot as plt

from ..plotter import Plotter


class AtlasMixin:

    def checkpoint_plot_state(self, gmodel, include_samples=True):
        """Keep the last completed fit, independent of later partial updates."""
        self._plot_checkpoint = {
            'gmodel': gmodel, 'df': self.df.copy(deep=True),
            'df_params': self.df_params.copy(deep=True),
            'sampler': self.sampler if include_samples else None,
            'resampled': self.resampled if include_samples else None,
            'burnin': self.burnin, 'thin': self.thin,
        }

    def make_error_atlas(self):
        state = getattr(self, '_plot_checkpoint', None)
        if state is None:
            print('[plot] No completed fit available for failure diagnostics', flush=True)
            return
        original = {key: getattr(self, key) for key in state}
        try:
            for key, value in state.items():
                setattr(self, key, value)
            self.make_atlas(gmodel=state['gmodel'], error=True)
        finally:
            for key, value in original.items():
                setattr(self, key, value)

    def make_atlas(self, gmodel=None, cleanup=False, intermediate=False, error=False) -> None:
        if intermediate and not getattr(self, 'intermediate_plots', False):
            return
        if self.path_plot is None:
            return
        
        if gmodel is None:
            gmodel = self.gmodel if self.gmodel is not None else self.gmodel_1G
        if gmodel is None:
            return
        
        plotter = None
        try:
            plotter = Plotter(
                self.path_plot,
                self.name_cube,
                self.suffix,
                self.list_disp,
                self.df,
                self.df_params,
                gmodel,
                self.sampler,
                self.resampled,
                self.burnin,
                self.thin,
                self.list_NHI_
            )
            plotter.walk_plot_max_steps = getattr(self, 'walk_plot_max_steps', 1000)
            plotter.walk_plot_max_walkers = getattr(self, 'walk_plot_max_walkers', 30)
            if error:
                plotter.savename_atlas = plotter.savename_atlas.with_name(
                    plotter.savename_atlas.stem + '_error.png')
            # Corner images are derived from the current state, never an older run.
            for path in (plotter.savename_corner_emcee, plotter.savename_corner_resample):
                path.unlink(missing_ok=True)
        
            if 'S31' in gmodel.names_param: G=3
            if 'S21' in gmodel.names_param: G=2
            if 'S1'  in gmodel.names_param: G=1
        
            # plotter.makeplot_atlas(G)
                        
            try:
                plotter.makeplot_atlas(G)
            except Exception as e:
                print(self.name_cube, self.suffix, e)
                # raise
        
            if cleanup: plotter.cleanup()
            
        
        finally:
            plotter = None
            plt.close('all')
            gc.collect()
        return
    
