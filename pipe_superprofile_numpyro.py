import glob
import multiprocessing
import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from natsort import natsorted
from plotfit import Plotfit
from shiftnstack import ShiftnStack
import datetime
from pprint import pprint
from tqdm import tqdm
from threading import Thread, Event

warnings.filterwarnings("ignore", category=RuntimeWarning)
matplotlib.use('Agg')

def task_clfy(name_cube):

    if(os.path.exists(dict_glob[name_cube]['path_clfy']) and overwrite_classify==False): pass
    else:
        from class_baypi_classify import Classify
        if os.path.exists(dict_glob[name_cube]['path_cube'].parent/'segmts')==False: return
        classify = Classify(dict_glob[name_cube]['path_cube'],path_baygaud=os.getenv('BAY'))
        classify.run(snlim_peak=snlim_peak, key_classify=key_classify, remove_coolwarmhot=True)
        del classify    
    
    return

def task_main(name_cube):
    
    path_cube = dict_glob[name_cube]['path_cube']
    path_clfy = dict_glob[name_cube]['path_clfy']
    path_mask = dict_glob[name_cube]['path_mask']
    path_plot = dict_glob['path_plot']
    path_temp = dict_glob['path_temp']
    suffix = dict_glob['suffix']
    
    path_stacked = path_cube.parent / f'stacked_df{suffix}.csv'
    path_stacked_disps = path_cube.parent / f'stacked_list_disps{suffix}.npy'
    
    if os.path.exists(path_cube)==False: return
    if os.path.exists(path_clfy)==False: return
    
    # if os.path.exists(path_stacked) and os.path.exists(path_stacked_disps):
    #     df_stacked = pd.read_csv(path_stacked, sep='\s+')
    #     list_disps = np.load(path_stacked_disps)
        
    # else:
    #     sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask,
    #                     filter_genuine=True)
    #     sns.run()
        
    #     sns.save_df_stacked(path_stacked)
    #     sns.save_list_disps(path_stacked_disps)
        
    #     df_stacked = sns.df_stacked
    #     list_disps = sns.list_disps
        
    #     del sns
    
    sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask,
                        filter_genuine=True)
    sns.run()
    
    # sns.save_df_stacked(path_stacked)
    # sns.save_list_disps(path_stacked_disps)
    
    df_stacked = sns.df_stacked
    dict_disp  = sns.dict_disp
        
    plotfit = Plotfit(df_stacked, dict_disp=dict_disp, name_cube=name_cube,
                        path_plot=path_plot,
                        path_temp=path_temp,
                        plot_autocorr=True,
                        longest_length=dict_glob['longest_length'],
                        maxtaskperchild=dict_glob['maxtaskperchild']
                        )

    plotfit.dict_params = dict_glob['dict_params']
    # if name_cube=='VCC1727' and dict_glob['suffix']=='_O1.0':
    #     print(f'[Pipe_superprofile] {name_cube}{dict_glob["suffix"]} exception rule applied: B2=free -> fix')
    #     plotfit.dict_params['B2']='fix'
    
    plotfit.run(suffix=suffix, 
                pbar_resample=dict_glob['pbar_resample'],
                # nsample_resample=10) #for debug
                # nsample_resample=399) #p=0.05
                nsample_resample=1499) #p=0.01
                # nsample_resample=10000) 

    plotfit.df.to_string(       dict_glob['path_temp']/'{}_plotfitter_temp_gfit.txt'.format(name_cube))
    plotfit.df_params.to_string(dict_glob['path_temp']/'{}_plotfitter_temp_para.txt'.format(name_cube))
    del plotfit

def multirun_clfy(names_cube, num_cores=1):
    
    print('[Pipe_superprofile] BAYGAUD classify start')
    
    pbar = tqdm(names_cube)
    lists = np.array(names_cube, dtype=object)
    pool = multiprocessing.Pool(processes=num_cores)
    
    with tqdm(total=len(lists), leave=True) as pbar:
        for _ in pool.imap_unordered(task_clfy, lists):
            pbar.update()
            
    print('[Pipe_superprofile] BAYGAUD classify end')

    pool.close()
    pool.join()
    return

def periodic_message(interval, stop_event):
    while not stop_event.is_set():
        time.sleep(interval)
        try: 
        
            texts_stat = natsorted(glob.glob(str(dict_glob['path_temp'] / 'stat*.txt')))
            results = []
            
            for filepath in texts_stat:
                with open(filepath, 'r') as file:
                    splits = file.readline().strip().split('.')
                    results.append((splits[0], splits[1], splits[2], splits[3]))  # name, status, iteration
            
            names_cube, stats, iters, etas = zip(*results)
            
            df_stat = pd.DataFrame({
                'Name': names_cube,
                'Status': stats,
                'Iteration': iters,
                'ETA': etas
            })
        except: continue
        
        # print(df_stat.to_string())
        df_stat.to_string(dict_glob['path_output']/'Plotfit_stat.csv', index=False)

def multirun_main(names_cube, num_cores=1):
    
    if len(names_cube)==1: dict_glob['pbar_resample']=True
    
    longest_string = max(names_cube, key=len)
    longest_length = len(longest_string)
    dict_glob['longest_length'] = longest_length
    
    commdir = dict_glob[names_cube[0]]['path_cube'].parent.parent
    seed    = dict_glob['seed']
    suffix  = dict_glob['suffix']
    
    tempdir = dict_glob['path_output']/f'temp_{seed}'
    dict_glob['path_temp'] = tempdir
    if(os.path.exists(tempdir)):
        shutil.rmtree(tempdir)
    os.mkdir(tempdir)
    
    plotdir = dict_glob['path_output']/'png'
    dict_glob['path_plot'] = plotdir
    if os.path.exists(plotdir)==False:
        os.mkdir(plotdir)
    
    savename_gfit = dict_glob['path_output']/'info_stacked{}_GFIT.csv'.format(suffix)
    savename_para = dict_glob['path_output']/'info_stacked{}_GFIT_params.csv'.format(suffix)
    
    print('[Pipe_superprofile] GFIT start')
    print('       Path origin: {}'.format(commdir))
    print('       Path output: {}'.format(dict_glob['path_output']))
    print('            suffix: {}'.format(suffix))
    # print(savename_gfit)
    # print(savename_para)
    maxtaskperchild=np.max([int(num_cores/len(names_cube)),1]).item()
    print('maxtaskperchild:{}'.format(maxtaskperchild))
    
    dict_glob['maxtaskperchild'] = maxtaskperchild
    
    pprint(dict_glob['dict_params'])
    
    pbar = tqdm(names_cube)
    lists = np.array(names_cube, dtype=object)
    pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=maxtaskperchild)
    
    interval = 1
    stop_event = Event()
    timer_thread = Thread(target=periodic_message, args=(interval, stop_event))
    timer_thread.start()
    
    with tqdm(total=len(lists), leave=True, desc=suffix) as pbar:
        for _ in pool.imap_unordered(task_main, lists):
            pbar.update()

    pool.close()
    pool.join()
    
    stop_event.set()
    timer_thread.join()
    
    temps = natsorted(np.array(glob.glob(str(tempdir/"*_plotfitter_temp_gfit.txt"))))
    df = pd.DataFrame()
    for i, temp in enumerate(temps):        
        df_temp = pd.read_csv(temp, sep='\s+')
        df = pd.concat([df, df_temp])
    
    df = df.reset_index(drop=True)
    for column in df.columns:
        if(column=='Name'):
            continue
        else:
            df[column] = pd.to_numeric(df[column])
        
    pd.options.display.float_format = '{:.3E}'.format
    print(df)

    for temp in temps:
        os.remove(temp)

    df.to_string(savename_gfit, index=False)
    
    temps = natsorted(np.array(glob.glob(str(tempdir/"*_plotfitter_temp_para.txt"))))
    df = pd.DataFrame()
    for i, temp in enumerate(temps):        
        df_temp = pd.read_csv(temp, sep='\s+')
        df = pd.concat([df, df_temp])
    
    df = df.reset_index(drop=True)
    for column in df.columns:
        if(column=='Name' or column=='Reliable'):
            continue
        else:
            df[column] = pd.to_numeric(df[column])
        
    pd.options.display.float_format = '{:.3E}'.format
    print(df)

    for temp in temps:
        os.remove(temp)

    df.to_string(savename_para, index=False)
    
    
    return

homedirs = [
    # '/home/mskim/workspace/research/data/spec_interp/10.3kms/AVID',
    # '/home/mskim/workspace/research/data/spec_interp/10.3kms/LITTLE_THINGS',
    # '/home/mskim/workspace/research/data/spec_interp/10.3kms/THINGS',
    # '/home/mskim/workspace/research/data/spec_interp/10.3kms/VIVA',
    # '/home/mskim/workspace/research/data/spec_interp/10.3kms/VLA-ANGST',
    
    # '/home/mskim/workspace/research/data/spec_interp/5.16kms/AVID',
    # '/home/mskim/workspace/research/data/spec_interp/5.16kms/LITTLE_THINGS',
    # '/home/mskim/workspace/research/data/spec_interp/5.16kms/THINGS',
    # '/home/mskim/workspace/research/data/spec_interp/5.16kms/VLA-ANGST',
    
    # '/home/mskim/workspace/research/data/test_resolution/10arcsec',
    # '/home/mskim/workspace/research/data/test_resolution/20arcsec',
    # '/home/mskim/workspace/research/data/test_resolution/30arcsec'
    
    '/home/mandu/workspace/research/data/spec_interp/10.3kms/VLA-ANGST'
]

nametype_cube = 'cube.fits'
nametype_galaxy = '*181'
# nametype_galaxy = 'DDO70'

# multipliers = [0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0, 0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.4,1.5,1.7,1.8,1.9]#, 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
# multipliers = [0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0]
# multipliers = [1.0]

# multipliers = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

# multipliers = [1.2,1.6,2.0, 0.3,0.5,0.7,0.9,1.1,1.3,1.4,1.5,1.7,1.8,1.9, 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
# multipliers = [2.9,3.0]

snlim_peak = 3.0
key_classify = '3'

num_cores = 20

remove_temp = True
overwrite_classify = False
bool_do_clfy  = True

bool_overwrite = 1

bool_do_whole  = 1
bool_do_inner  = 0
bool_do_outer  = 0

bool_pack_output = 1

dict_glob = {}
dict_glob['dict_params']={
    'A21':'free', # free only
    'A22':'free', # free only
    'V21':'fix',    # fix will make V21=V1; 0 will fix velocity to zero
    'V22':'fix',  # fix will make V22=V21, has to be 'fix' if V21=fix or V21=0
    'S21':'free', # free only
    'S22':'free', # free only
    'B2' :'free'   # fix will make B1=B2s
}
statV21 = 'r' if dict_glob['dict_params']['V21']=='free' else '0' if dict_glob['dict_params']['V21']=='0' else 'x'
statV22 = 'r' if dict_glob['dict_params']['V22']=='free' else 'x'
statB2  = 'r' if dict_glob['dict_params']['B2' ]=='free' else 'x'
dict_glob['pbar_resample'] = False

timei = time.time()

for homedir in homedirs:
    
    homedir = Path(homedir)
    path_output = None
    # path_output = homedir.parent / f'V21{statV21}V22{statV22}B2{statB2}_{homedir.name}'
    
    paths_cube = glob.glob(str(homedir / f'{nametype_galaxy}/{nametype_cube}'))
    names_cube = []
    for path_cube in paths_cube:        
        path_cube = Path(path_cube)
        wdir = path_cube.parent 
        name_cube = wdir.name
        
        if os.path.exists(wdir/'segmts')==False: continue
        
        names_cube = np.append(names_cube, name_cube)
        
        dict_glob[name_cube]={
            'path_cube':path_cube,
            'name_cube':name_cube,
            'path_clfy':wdir/'segmts_merged_n_classified.{}'.format(key_classify),
            'path_mask':None
        }
    
    dict_glob['suffix'] = ''
    dict_glob['seed'] = str(time.time()).split(".")[-1]
    
    if path_output is None: path_output=homedir
    dict_glob['path_output'] = path_output
    if os.path.exists(path_output)==False: os.mkdir(path_output)
    
    temps_pre = glob.glob(str(path_output/'temp_*/'))
    for temp_pre in temps_pre:
        shutil.rmtree(temp_pre)
    
    if bool_overwrite:
        outputs_pre = glob.glob(str(path_output/'info_stacked*'))
        for output_pre in outputs_pre:
            print('[Pipe_superprofile] Removing pre-existing fit data')
            os.remove(output_pre)
        
        if os.path.exists(path_output/'png'):
            print('[Pipe_superprofile] Removing pre-existing figure directory')
            shutil.rmtree(path_output/'png')
            os.mkdir(path_output/'png')
    
    if bool_do_clfy:
        multirun_clfy(names_cube, num_cores=num_cores)

    if bool_do_whole:
        if os.path.exists(path_output/'info_stacked_GFIT.csv'): pass
        else: multirun_main(names_cube, num_cores=num_cores)
    
    if bool_do_inner:
    
        for multiplier in multipliers:
            
            dict_glob['suffix'] = '_I{}'.format(multiplier)
            
            if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
                pass
            else:
                from run_makemask_ellipse import run_makemask as run_makemask_ellipse
                run_makemask_ellipse(paths_cube, multiplier_radius=multiplier, path_df='/home/mskim/workspace/research/data/cat_diameters.csv')
                for name_cube in names_cube:
                    dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask{}.fits'.format(dict_glob['suffix'])
                multirun_main(names_cube, num_cores=num_cores)
    
    if bool_do_outer:
        for multiplier in multipliers:
            
            dict_glob['suffix'] = '_O{}'.format(multiplier)
            if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
                pass
            else:
                for name_cube in names_cube:
                    dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask{}.fits'.format(dict_glob['suffix'])
                multirun_main(names_cube, num_cores=num_cores)
            
    print('[Pipe_superprofile] Finished for {}'.format(path_output))
    
    if bool_pack_output:
        print('[Pipe_superprofile] Packing up outputs')
        os.chdir(path_output)
        files_to_tar = ['png'] + glob.glob('info_stacked*')
        from subprocess import DEVNULL, STDOUT, check_call
        if os.path.exists('outputs.tar.gz'):
            os.remove('outputs.tar.gz')
        check_call(['tar', 'cvzf', 'outputs.tar.gz'.format(statV21,statV22,statB2)] + files_to_tar, stdout=DEVNULL, stderr=STDOUT)
        
timef = time.time()

print('[Pipe_superprofile] Done!')
print('                    Total time elapsed: {}'.format(str(datetime.timedelta(seconds=timef-timei))))