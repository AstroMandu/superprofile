# paths_cube = natsorted(list(Path('/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam').glob('*43/cube.fits')))

from astropy import units as u
from pathlib import Path

path_cube = Path('/home/mandu/workspace/research/tools/tool_superprofile/pack_superprofile/test/NGC2903/cube.fits')

galname = path_cube.parent.name
print(galname)

# fig, axs = plt.subplots(ncols=2, sharey=True)

from shiftnstack import ShiftnStack
path_clfy = path_cube.parent/'segmts_merged_n_classified.3'
sns = ShiftnStack(path_cube, path_clfy)#, path_mask='/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO43/mask_I0.8.fits')
sns.pbar = True
sns.run()

from plotfit import Plotfit

plotfit = Plotfit(sns.df_stacked, dict_disp=sns.dict_disp, name_cube=galname,
                    path_plot='/home/mandu/workspace/research/tools/tool_superprofile/pack_superprofile/test/png',
                    path_temp='/home/mandu/workspace/research/tools/tool_superprofile/pack_superprofile/test/temp',
                    plot_autocorr=False,
                    longest_length=10,
                    vdisp_low_intrinsic=5*(u.km/u.s),
                    )

plotfit.dict_params = {
    'A21':'free', # free only
    'A22':'free', # free only
    'V21':'fix',    # fix will make V21=V1; 0 will fix velocity to zero
    'V22':'fix',  # fix will make V22=V21, has to be 'fix' if V21=fix or V21=0
    'S21':'free', # free only
    'S22':'free', # free only
    'B2' :'free'   # fix will make B1=B2s
}

plotfit.maxiter_2G = 10000
plotfit.maxtaskperchild=10

plotfit.run(suffix='_', 
            pbar_resample=True,
            # nsample_resample=10) #for debug
            nsample_resample=399) #p=0.05
            # nsample_resample=399) #p=0.01
            # nsample_resample=10000) 