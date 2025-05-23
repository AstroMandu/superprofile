
# from .shiftnstack.class_ShiftnStack_baygaudpi import ShiftnStack
# from .plotfit.class_Plotfitter_emcee import Plotfit

from plotfit import Plotfit
from shiftnstack import ShiftnStack, Filter

import pandas as pd
# df_stacked = pd.read_csv('/home/mandu/workspace/research/data/spec_interp/10.3kms/test_Field/WLM/stacked_total_I1.0.csv', sep='\s+')



import os
from pathlib import Path

path_cube = Path('/home/mandu/workspace/research/data/spec_interp/10.3kms/test_Field/CVnIdwA/cube.fits')
path_clfy = Path('/home/mandu/workspace/research/data/spec_interp/10.3kms/test_Field/CVnIdwA/segmts_merged_n_classified.2')
path_mask = Path('/home/mandu/workspace/research/data/spec_interp/10.3kms/test_Field/CVnIdwA/mask_stack.fits')


path_plot = path_cube.parent.parent/'png'
name_cube = path_cube.parent

if(os.path.exists(path_plot)==False): os.mkdir(path_plot)

sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask,
                    filter_genuine=True)
sns.run()

plotfit = Plotfit(sns.df_stacked, list_disps=sns.list_disps, name_cube=name_cube,
                    path_plot=path_plot,
                    plot_autocorr=True,
                    )
del sns
plotfit.run(makeplot=True, suffix='')

print(plotfit.df)
print(plotfit.df_params)