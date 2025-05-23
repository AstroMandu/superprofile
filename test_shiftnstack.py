import pylab as plt
from pathlib import Path
from natsort import natsorted


paths_cube = natsorted(list(Path('/home/mandu/workspace/research/data/THINGS_halfbeam').glob('NGC4449/cube.fits')))

    
for path_cube in paths_cube:
    
    print(path_cube)
    
    galname = path_cube.parent.name
    print(galname)
    
    fig, axs = plt.subplots(ncols=2, sharey=True)
    
    from shiftnstack import ShiftnStack
    path_clfy = path_cube.parent/'segmts_merged_n_classified.3'
    sns = ShiftnStack(path_cube, path_clfy, bgsub=False)#, path_mask='/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO43/mask_I0.8.fits')
    sns.pbar = True
    sns.run()
    
    df = sns.df_stacked
    axs[0].plot(df['x'],df['y'], color='gray')
    
    
    sns = ShiftnStack(path_cube, path_clfy, path_vf_secondary=path_cube.parent/'cube_mom1.fits')#, path_mask='/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO43/mask_I0.8.fits')
    sns.pbar = True
    sns.run(stack_secondary=True)
    
    # from shiftnstack import ShiftnStack_hermite as ShiftnStack
    # path_hermite = path_cube.parent/'hermite.npy'
    # sns = ShiftnStack(path_cube, path_hermite)#, path_mask='/home/mandu/workspace/research/data/LITTLE_THINGS_halfbeam/DDO43/mask_I0.8.fits')#, path_mask=path_clfy/'sgfit/sgfit.G3_1.2.fits')
    # sns.pbar = True
    # sns.run()
    
    df = sns.df_stacked
    axs[1].plot(df['x'],df['y'], color='gray')
    
    plt.show()
    
    