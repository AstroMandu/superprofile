import numpy as np
from run_makemask_ellipse import run_makemask as run_makemask_ellipse
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
from pathlib import Path
import multiprocessing
from astropy import units as u
from shiftnstack import ShiftnStack, ShiftnStack_hermite


def job(path_cube):
    
    name_cube = path_cube.parent.name
    path_clfy = path_cube.parent/'segmts_merged_n_classified.3'
    
    df = pd.DataFrame()
    
    for i, multiplier in enumerate(multipliers):
        
        suffix = f'_I{multiplier}r25'
        
        path_mask = path_cube.parent/f'mask{suffix}.fits'
        
        sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask)
        if name_cube=='NGC1569':
            sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))      
        # sns.pbar = True
        sns.run()
        
        df_stacked = sns.df_stacked
        df_stacked[suffix[1:]] = df_stacked['y']
        
        if i==0: df = df_stacked[['x',suffix[1:]]]
        else:
            df = pd.merge(df, df_stacked[['x',suffix[1:]]])
            
    path_df_out = path_cube.parent/'df_stacked.csv'
    df.to_string(path_df_out, index=False)
    
    path_clfy = path_cube.parent/'hermite.npy'
    path_df_out = path_cube.parent/'df_stacked.csv'
    
    # df = pd.DataFrame()
    df = pd.read_csv(path_df_out, sep='\s+')
    
    for i, multiplier in enumerate(multipliers):
        
        suffix = f'_I{multiplier}r25'
        path_mask = path_cube.parent/f'mask{suffix}.fits'
        
        suffix = f'_I{multiplier}r25_her'
        
        sns = ShiftnStack_hermite(path_cube, path_clfy, path_mask=path_mask)
        if name_cube=='NGC1569':
            sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))      
        # sns.pbar = True
        sns.run()
        
        df_stacked = sns.df_stacked
        df_stacked[suffix[1:]] = df_stacked['y']
        
        # df = pd.merge(df, df_stacked[['x',suffix[1:]]], on='x', how='left')
        
        df[suffix[1:]] = df_stacked['y']
        
    df.to_string(path_df_out, index=False)
    
    

multipliers = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# multipliers = [1.0]

for survey in ['LITTLE_THINGS','THINGS','VLA-ANGST','AVID','VIVA']:
# for survey in ['VIVA']:

    paths_cube = natsorted(list(Path(f'/home/mandu/workspace/research/data/{survey}_halfbeam').glob('*/cube.fits')))
    
    # for path_cube in paths_cube:
    #     print(path_cube.parent.name)
    #     job(path_cube)
    
    pbar = tqdm(paths_cube)
    lists = np.array(paths_cube, dtype=object)
    pool = multiprocessing.Pool(processes=15)
    
    with tqdm(total=len(lists), leave=True) as pbar:
        for _ in pool.imap_unordered(job, lists):
            pbar.update()
        
    pool.close()
    pool.join()