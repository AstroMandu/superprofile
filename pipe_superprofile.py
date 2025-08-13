import datetime
import glob
import multiprocessing
import os
import shutil
import time
import warnings
from pathlib import Path
from pprint import pprint
from threading import Event, Thread

import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
from natsort import natsorted
from plotfit import Plotfit
from run_makemask_ellipse import run_makemask as run_makemask_ellipse
from run_makemask_ring    import run_makemask as run_makemask_ring
from makemask_angle import makemask as makemask_angle
from shiftnstack import ShiftnStack, ShiftnStack_hermite
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
matplotlib.use('Agg')


def task_clfy(name_cube):

    if(os.path.exists(dict_glob[name_cube]['path_clfy']) and overwrite_classify==False): pass
    else:
        from class_baypi_classify import Classify
        if os.path.exists(dict_glob[name_cube]['path_cube'].parent/'segmts')==False: return
        classify = Classify(dict_glob[name_cube]['path_cube'],path_baygaud=os.getenv('BAY'),vdisp_low_intrinsic=dict_glob['vdisp_low_intrinsic'])
        classify.run(snlim_peak=snlim_peak, key_classify=key_classify, remove_coolwarmhot=True)
        del classify    
    
    return

def task_main(name_cube):
    
    path_cube = dict_glob[name_cube]['path_cube']
    path_mask = dict_glob[name_cube]['path_mask']
    
    if use_secondary_vf:
        path_vf_secondary = dict_glob[name_cube]['path_vf_secondary']
    else:
        path_vf_secondary = None
    
    mode = dict_glob['mode']
    
    if os.path.exists(path_cube)==False: return
    
    if mode=='baygaud':
        path_clfy = dict_glob[name_cube]['path_clfy']
        
        if os.path.exists(path_clfy)==False: return
        if path_vf_secondary is not None:
            if os.path.exists(path_vf_secondary)==False: path_vf_secondary=None
        
        sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask, path_vf_secondary=path_vf_secondary)
        
    if mode=='hermite':
        path_hermite   = dict_glob[name_cube]['path_hermite']
        if os.path.exists(path_hermite)==False: return
        
        sns = ShiftnStack_hermite(path_cube, path_hermite, path_mask=path_mask, path_vf_secondary=path_vf_secondary)

    if name_cube=='NGC1569':
        sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))
    
    sns.run(stack_secondary=True)
    
    if path_mask is not None:
        if os.path.exists(path_mask): os.remove(path_mask)
    
    path_plot = dict_glob['path_plot']
    path_temp = dict_glob['path_temp']
    
    df_stacked = sns.df_stacked
    dict_disp  = sns.dict_disp
    
    df_stacked.to_string(path_cube.parent/'df_stacked.csv', index=False)
        
    plotfit = Plotfit(df_stacked, dict_disp=dict_disp, name_cube=name_cube,
                        path_plot=path_plot,
                        path_temp=path_temp,
                        plot_autocorr=False,
                        longest_length=dict_glob['longest_length'],
                        vdisp_low_intrinsic=dict_glob['vdisp_low_intrinsic']
                        )
    
    if truth_from_resampling: plotfit.truth_from_resampling = True

    plotfit.dict_params = dict_glob['dict_params']
    
    plotfit.run(suffix=dict_glob['suffix'], 
                pbar_resample=dict_glob['pbar_resample'],
                # nsample_resample=10) #for debug
                # nsample_resample=399) #p=0.05
                nsample_resample=dict_glob['nsample_resample']) #p=0.01
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
                    splits = file.readline().strip().split()
                    results.append((splits[0], splits[1], splits[2], splits[3]))  # name, status, iteration
            
            names_cube, stats, iters, etas = zip(*results)
            
            df_stat = pd.DataFrame({
                'Name': names_cube,
                'Status': stats,
                'Iteration': iters,
                'ETA': etas
            })
        except Exception as e: 
            print(e)
            continue
        
        df_stat.to_string(dict_glob['path_output']/'Plotfit_stat_{}.csv'.format(dict_glob['seed']), index=False)

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

    df.to_string(savename_para, index=False)
    
    for temp in temps:
        os.remove(temp)
        

    
    return

def do_inner(multiplier):
        
    dict_glob['suffix'] = suffix+'I{}{}'.format(multiplier,dict_glob['radtag'])
    masksuffix = '_I{}{}'.format(multiplier,dict_glob['radtag'])
    run_makemask_ellipse(paths_cube, multiplier_radius=multiplier, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')

    if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
    # if False:
        pass
    else:
        
        for name_cube in names_cube:
            dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(masksuffix)
        if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
        multirun_main(names_cube, num_cores=num_cores)
            
    return

def do_outer(multiplier):
        
    dict_glob['suffix'] = suffix+'O{}{}'.format(multiplier,dict_glob['radtag'])
    masksuffix = '_O{}{}'.format(multiplier,dict_glob['radtag'])
    run_makemask_ellipse(paths_cube, multiplier_radius=multiplier, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
    
    # # if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
    # if False:
    #     pass
    # else:
    #     for name_cube in names_cube:
    #         dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(masksuffix)
    #     if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    #     multirun_main(names_cube, num_cores=num_cores)
    
    return

def do_rings(multiplier):
    
    dict_glob['suffix'] = suffix+'R{}{}'.format(multiplier,dict_glob['radtag'])
    if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    masksuffix = '_R{}{}'.format(multiplier,dict_glob['radtag'])
    run_makemask_ring(paths_cube, multiplier_radius_center=multiplier, width='beam', col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
    
    if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
        pass
    else:
        for name_cube in names_cube:
            dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(masksuffix)
        multirun_main(names_cube, num_cores=num_cores)
        
    return

def do_angles(angle, angle_width):
            
    dict_glob['suffix'] = suffix+'A{:0>3}_W{:0>3}'.format(angle,angle_width)
    if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    masksuffix = '_A{:0>3}_W{:0>3}'.format(angle,angle_width)
    if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
        pass
    else:
        for path_cube in paths_cube:
            
            makemask_angle(path_cube, angle_center=angle, angle_width=angle_width, 
                        path_df=path_data/'catalog/cat_diameters.csv',
            )
        for name_cube in names_cube:
            dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(masksuffix)
        
        multirun_main(names_cube, num_cores=num_cores)
    return
        
def do_angles_O05(angle,angle_width):
    run_makemask_ellipse(paths_cube, multiplier_radius=0.5, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
    
    dict_glob['suffix'] = suffix+'A{:0>3}_W{:0>3}_O0.5r25'.format(angle,angle_width)
    if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
        pass
    else:
        for path_cube in paths_cube:
            
            makemask_angle(path_cube, angle_center=angle, angle_width=angle_width, 
                        path_df=path_data/'catalog/cat_diameters.csv',
                        path_mask=path_cube.parent/'mask/mask_O0.5r25.fits',
                        savename=path_cube.parent/'mask/mask{}.fits'.format(dict_glob['suffix']))
        for name_cube in names_cube:
            dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(dict_glob['suffix'])
        
        multirun_main(names_cube, num_cores=num_cores)
    return
    
def do_angles_O10(angle,angle_width):
    run_makemask_ellipse(paths_cube, multiplier_radius=1.0, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
    
    dict_glob['suffix'] = suffix+'A{:0>3}_W{:0>3}_O1.0r25'.format(angle,angle_width)
    if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    if os.path.exists(path_output/'info_stacked{}_GFIT.csv'.format(dict_glob['suffix'])):
        pass
    else:
        for path_cube in paths_cube:
            
            makemask_angle(path_cube, angle_center=angle, angle_width=angle_width, 
                        path_df=path_data/'catalog/cat_diameters.csv',
                        path_mask=path_cube.parent/'mask/mask_O1.0r25.fits',
                        savename=path_cube.parent/'mask/mask{}.fits'.format(dict_glob['suffix']))
        for name_cube in names_cube:
            dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent/'mask/mask{}.fits'.format(dict_glob['suffix'])
        
        multirun_main(names_cube, num_cores=num_cores)
        
    return


path_data = Path('/home/mskim/workspace/research/data')
homedirs = [
    # path_data/'LITTLE_THINGS_halfbeam',
    # path_data/'THINGS_halfbeam',
    # path_data/'VLA-ANGST_halfbeam',
    # path_data/'AVID_halfbeam',
    # path_data/'VIVA_halfbeam',
    
    # path_data/'LITTLE_THINGS',
    # path_data/'AVID',
    # path_data/'VIVA',
    
    # path_data+'/Rory/HIcubefiles/tests',
    # path_data+'/test_resolution/10arcsec',
    # path_data+'/test_resolution/20arcsec',
    # path_data/'test_resolution/30arcsec',
    # path_data/'test_chanres/2.58kms',
    # path_data/'test_chanres/5.16kms',
    # path_data/'test_chanres/10.3kms',
    # path_data+'/Rory/RPfiles',
    # '/home/mskim/workspace/research/data/AVID'
    # '/home/mskim/workspace/research/data/AVID_halfbeam'
    
    '/home/mskim/workspace/research/data/test'
    # '/home/mskim/workspace/research/data/test_LBFGSB'
    # '/home/mskim/workspace/research/data/test_neldermead'
    # '/home/mskim/workspace/research/data/test_SNR3_snch_sbbw'
]

nametype_cube = 'cube.fits'
nametype_galaxy = '*'
# nametype_galaxy = '15,2,0.4'
# nametype_galaxy = 'DDO70'

# multipliers = [0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0, 0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.4,1.5,1.7,1.8,1.9]#, 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
# multipliers = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
# multipliers = [round(x, 2) for x in np.arange(0.10, 1.50 + 0.001, 0.05)]
multipliers = [round(x, 2) for x in np.arange(0.10, 1.50 + 0.001, 0.1)]

# multipliers = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# multipliers = [0.4]

# multipliers = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

# multipliers = [1.2,1.6,2.0, 0.3,0.5,0.7,0.9,1.1,1.3,1.4,1.5,1.7,1.8,1.9, 2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
# multipliers = [2.9,3.0]

# multipliers = multipliers[::2]
# multipliers = np.flip(multipliers[1::2])
# multipliers = np.flip(multipliers)

# col_radius = 'r25';      radtag='r25'
col_radius = 'RHI(kpc)'; radtag='RHI'

suffix = '_'

snlim_peak    = 2
key_classify = '2'
# suffix  += 'SNR2.5_'

use_secondary_vf = True; #suffix+='VFsec_'
# use_secondary_vf = False

truth_from_resampling = True

num_cores = 33

bool_overwrite     = 1
remove_temp        = 1

overwrite_classify = 1
bool_do_clfy       = 1

bool_do_whole  = 0
bool_do_inner  = 0
bool_do_outer  = 0
bool_do_rings  = 1

bool_do_angles = 0
angles=[0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]

# angles = angles[::2]
# angles = np.flip(angles[1::2])
# angles = np.flip(angles)

bool_pack_output = 0

# nsample_resample = 10 # p~0.05
# nsample_resample = 399 # p~0.05
nsample_resample = 1499 # p~0.01

dict_glob = {}
dict_glob['dict_params']={
    'A21':'free', # free only
    'A22':'free', # free only
    'V21':'fix',    # fix will make V21=V1; 0 will fix velocity to zero
    'V22':'fix',  # fix will make V22=V21, has to be 'fix' if V21=fix or V21=0
    'S21':'free', # free only
    'S22':'free', # free only
    'B2' :'fix'   # fix will make B1=B2s
}
statV21 = 'r' if dict_glob['dict_params']['V21']=='free' else '0' if dict_glob['dict_params']['V21']=='0' else 'x'
statV22 = 'r' if dict_glob['dict_params']['V22']=='free' else 'x'
statB2  = 'r' if dict_glob['dict_params']['B2' ]=='free' else 'x'
dict_glob['pbar_resample'] = False
dict_glob['nsample_resample'] = nsample_resample
dict_glob['mode'] = 'baygaud'
# dict_glob['mode'] = 'hermite'
# dict_glob['vdisp_low_intrinsic'] = 1.5
dict_glob['vdisp_low_intrinsic'] = 0
dict_glob['radtag'] = radtag

timei = time.time()

for homedir in homedirs:
    
    homedir = Path(homedir)
    path_output = None
    # path_output = homedir.parent / f'V21{statV21}V22{statV22}B2{statB2}_{homedir.name}'
    
    # path_output = homedir.parent / f'{homedir.name}_2beam'
    # path_output = homedir.parent / f'{homedir.name}_2VFonly'
    
    paths_cube = glob.glob(str(homedir / f'{nametype_galaxy}/{nametype_cube}'))
    names_cube = []
    for i,path_cube in enumerate(paths_cube):
        path_cube = Path(path_cube)
        paths_cube[i] = path_cube
        wdir = path_cube.parent 
        name_cube = wdir.name
        
        if dict_glob['mode']=='baygaud':
            if os.path.exists(wdir/'segmts')==False: continue
        if dict_glob['mode']=='hermite':
            if os.path.exists(wdir/'hermite.npy')==False: continue
        
        names_cube = np.append(names_cube, name_cube)
        
        dict_glob[name_cube]={
            'path_cube':path_cube,
            'name_cube':name_cube,
            'path_mask':None
        }
        
        if dict_glob['mode']=='baygaud':
            dict_glob[name_cube]['path_clfy']=wdir/'segmts_merged_n_classified.{}'.format(key_classify)
        if dict_glob['mode']=='hermite':
            dict_glob[name_cube]['path_hermite']=wdir/'hermite.npy'
        dict_glob[name_cube]['path_vf_secondary']=wdir/'cube_mom1.fits'
    
    dict_glob['suffix'] = suffix
    if dict_glob['mode']=='hermite': dict_glob['suffix']+='_her' 
    dict_glob['seed'] = str(time.time()).split(".")[-1]
    
    if path_output is None: path_output=homedir
    dict_glob['path_output'] = path_output
    if os.path.exists(path_output)==False: os.mkdir(path_output)
    
    if remove_temp:
        
        temps_pre = glob.glob(str(path_output/'temp_*/'))
        for temp_pre in temps_pre: shutil.rmtree(temp_pre)
        temps_pre = glob.glob(str(path_output/'Plotfit_stat*.csv'))
        for temp_pre in temps_pre: os.remove(temp_pre)
    
    if bool_overwrite:
        outputs_pre = glob.glob(str(path_output/'info_stacked*'))
        for output_pre in outputs_pre:
            print('[Pipe_superprofile] Removing pre-existing fit data')
            os.remove(output_pre)
        
        if os.path.exists(path_output/'png'):
            print('[Pipe_superprofile] Removing pre-existing figure directory')
            shutil.rmtree(path_output/'png')
            os.mkdir(path_output/'png')
    
    if bool_do_clfy and dict_glob['mode']=='baygaud':
        multirun_clfy(names_cube, num_cores=num_cores)

    if bool_do_whole:
        if os.path.exists(path_output/'info_stacked_{}GFIT.csv'.format(dict_glob['suffix'])): pass
        else: multirun_main(names_cube, num_cores=num_cores)
    
    if bool_do_inner:
        for multiplier in multipliers: do_inner(multiplier)
    
    if bool_do_outer:
        for multiplier in multipliers: do_outer(multiplier)
                
    if bool_do_rings:
        for multiplier in multipliers: do_rings(multiplier)
                
    if bool_do_angles:
        for angle in angles: do_angles(angle,180)
        for angle in angles: do_angles_O05(angle,180)
        for angle in angles: do_angles_O10(angle,180)
        
        # for angle in angles: do_angles(angle,90)
        # for angle in angles: do_angles_O05(angle,90)
        # for angle in angles: do_angles_O10(angle,90)
        

    
    try:
        shutil.rmtree(dict_glob['path_output']/f'temp_{dict_glob["seed"]}')
        os.remove(dict_glob['path_output']/'Plotfit_stat_{}.csv'.format(dict_glob['seed']))
    except:
        pass
            
    print('[Pipe_superprofile] Finished for {}'.format(path_output))
    
    if bool_pack_output:
        print('[Pipe_superprofile] Packing up outputs')
        os.chdir(path_output)
        files_to_tar = ['png'] + glob.glob('info_stacked*')
        from subprocess import DEVNULL, STDOUT, check_call
        if os.path.exists('outputs.tar.gz'):
            os.remove('outputs.tar.gz')
        check_call(['tar', 'cvzf', 'outputs.tar.gz'] + files_to_tar, stdout=DEVNULL, stderr=STDOUT)
        
timef = time.time()

print('[Pipe_superprofile] Done!')
print('                    Total time elapsed: {}'.format(str(datetime.timedelta(seconds=timef-timei))))