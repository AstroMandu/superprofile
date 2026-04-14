# Expects on self: path_plot, name_cube, suffix, list_disp, list_NHI_,
#                  df, df_params, gmodel, gmodel_1G, sampler, resampled,
#                  burnin, thin
from __future__ import annotations

import gc

import matplotlib.pyplot as plt

from ..plotter import Plotter


class AtlasMixin:

    def make_atlas(self, gmodel=None, cleanup=False) -> None:
        if self.path_plot is None:
            return
        
        if gmodel      is None: gmodel = self.gmodel
        if self.gmodel is None: gmodel = self.gmodel_1G
        
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
            
        del plotter
        
        plt.close('all')
        gc.collect()
        return
    
