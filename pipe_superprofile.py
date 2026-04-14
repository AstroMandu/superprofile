# -*- coding: utf-8 -*0-
import datetime
import gc
import glob
import multiprocessing as mp
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from threading import Event, Thread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsort_keygen, natsorted
from tqdm import tqdm

# ---- LIMIT INTERNAL THREADS (MUST be before numpy/pandas imports) ----
# 1 thread per process for OpenMP/BLAS/NumExpr
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
# avoid dynamic growth; lower spin-wait overhead
os.environ.setdefault("MKL_DYNAMIC",          "FALSE")
os.environ.setdefault("OMP_WAIT_POLICY",      "PASSIVE")
# --------


@dataclass(frozen=True)
class MaskCtx:
    suffix: str
    radtag: str
    mode: str
    path_output: str  # str or Path is fine; choose str to be safe for pickling



warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
matplotlib.use('Agg')

dict_glob = {}

# =========================
# Config & Globals
# =========================
# path_data = Path('/media/cusped03/mskim/workspace/research/data')
path_data = Path('/home/mskim/workspace/research/data')
homedirs = [
    
    # path_data/'LITTLE_THINGS_REG',
    # path_data/'THINGS_REG',
    # path_data/'VLA-ANGST_REG',
    # path_data/'VIVA_REG',

    # path_data/'AVID'
    # path_data/'AVID_hann',
    path_data/'AVID_hann_REG',
    # path_data/'THINGS_REG',
    
    # path_data/'test',
    # path_data/'AVID_sofia1beam',
    # path_data/'AVID_sofia3beam',
    
    # path_data/'Rory/RPfiles_0.01mJy',
    # path_data/'Rory/RPfiles_0.4mJy',
]

nametype_cube   = 'cube.fits'
nametype_galaxy = '*VCC152'
# nametype_galaxy = None

galaxies = None
names_LT = [
    # 'CVnIdwA',
    # 'DDO43',
    # 'DDO46',
    # 'DDO47',
    # 'DDO50',
    # 'DDO52',
    # 'DDO53',
    # 'DDO63',
    # 'DDO69',
    # 'DDO70',
    # 'DDO75',
    # 'DDO87',
    # 'DDO101',
    # 'DDO126',
    # 'DDO133',
    # 'DDO154',
    # 'DDO155',
    # 'DDO165',
    # 'DDO167',
    # 'DDO168',
    # 'DDO187',
    # 'DDO210',
    # 'DDO216',
    # 'F564-V3',
    # 'Haro29',
    'Haro36',
    'IC10',
    'IC1613',
    'LGS3',
    'M81DWA',
    'Mrk178',
    'NGC1569',
    'NGC2366',
    'NGC3738',
    'NGC4163',
    'NGC4214',
    'SAGDIG',
    'UGC8508',
    'VIIZw403',
    'WLM',
]

names_AVID = [
    # 'AGC225847',
    # 'VCC130',
    # 'VCC152',
    # 'VCC169',
    # 'VCC309',
    # 'VCC322',
    # 'VCC328',
    # 'VCC329',
    # 'VCC334',
    # 'VCC340',
    # 'VCC379',
    # 'VCC381',
    # 'VCC566',
    'VCC613',
    'VCC656',
    'VCC667',
    'VCC693',
    'VCC697',
    'VCC699',
    'VCC1091',
    'VCC1411',
    'VCC1778',
    'VCC1992',
    'VCC2006',
    'VCC2034',
    'VCC2037',
]

# galaxies = names_LT
# galaxies = names_AVID
# galaxies = names_ANGST

# multipliers = [round(x, 2) for x in np.arange(0.10, 1.50 + 0.001, 0.05)]
multipliers = [0.6]
# col_radius = 'r25';      radtag='r25'
col_radius = 'RHI(kpc)'; radtag='RHI'
suffix = '_'

# snlim_peak   =  10000
# key_classify = '10000'

use_secondary_vf      = 1
truth_from_resampling = 1
num_threads           = 60

dict_glob['mode'] = 'baygaud'; suffix_path_output = '_2GFIT_BAYMOM_prior3'; snlim_peak=3; key_classify=3; use_secondary_vf=1;
# dict_glob['mode'] = 'baygaud'; suffix_path_output = '_3GFIT_BAYMOM_MED_dim2_fluxF31+F32>F33'; snlim_peak=3; key_classify=3; use_secondary_vf=1;
# dict_glob['mode'] = 'baygaud'; suffix_path_output = '_3GFIT_BAYMOM_MED_dim2_ampl'; snlim_peak=3; key_classify=3; use_secondary_vf=1;
# dict_glob['mode'] = 'baygaud'; suffix_path_output = '_3GFIT_BAYMOM_MED'; snlim_peak=3; key_classify=3; use_secondary_vf=1;
# 
# dict_glob['mode'] = 'hermite'; suffix_path_output = '_3GFIT_HERMOM'; use_secondary_vf=1;
# dict_glob['mode'] = 'hermite'; suffix_path_output = '_3GFIT_HER';    use_secondary_vf=0;

statistics = 'MAP'
# statistics = 'MEDIAN'

bool_overwrite     = 1
remove_temp        = 1

overwrite_classify = 0
bool_do_clfy       = 1

guess_from_whole  = 0
fallback_to_2gfit = 0

bool_do_whole  = 1
bool_do_inner  = 0
bool_do_outer  = 0
bool_do_rings  = 0
bool_do_angles = 1

# widths_angle = [180]
widths_angle = [60,120,180]
PAs = range(0,360,15)
# PAs = [300]

# angles = [0,45,90,135,180,225,270,305]
# angles = [300]

# angles = angles[::2]

bool_pack_output = 0
# nsample_resample = 100
# nsample_resample = 1499 # p~0.01
nsample_resample = 10000

dict_glob['pbar_resample']    = False
dict_glob['nsample_resample'] = nsample_resample
# dict_glob['mode'] = 'baygaud'
# dict_glob['mode'] = 'hermite'
dict_glob['vdisp_low_intrinsic'] = 0
dict_glob['radtag'] = radtag

# dict_glob['method_minimize'] = 'Powell'
dict_glob['method_minimize'] = 'Nelder-Mead'

dict_jobs = {}  # parent-only enqueue

# =========================
# Helpers
# =========================
def get_mp_ctx():
    # 항상 spawn
    # return mp.get_context("spawn")
    return mp.get_context("forkserver")

def get_executor(max_workers):
    # spawn 컨텍스트로 ProcessPoolExecutor 만들기
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=get_mp_ctx())

def add_job(name_cube, suffix, path_mask):
    index = len(dict_jobs)
    dict_jobs[index] = dict_glob[name_cube].copy()
    dict_jobs[index]['suffix']    = suffix
    dict_jobs[index]['path_mask'] = path_mask

# =========================
# Classify
# =========================
def task_clfy(job):
    # lazy import stays fine
    from class_baypi_classify import Classify

    name_cube         = job['name_cube']
    path_cube         = job['path_cube']
    path_clfy         = job['path_clfy']
    vdisp_low_intr    = job['vdisp_low_intrinsic']
    overwrite         = job['overwrite_classify']

    # skip if already classified (unless overwrite)
    if os.path.exists(path_clfy) and not overwrite:
        return name_cube

    # require segmts dir
    if not os.path.exists(path_cube.parent / 'segmts'):
        return name_cube

    classify = Classify(
        path_cube,
        path_baygaud=os.getenv('BAY'),
        vdisp_low_intrinsic=vdisp_low_intr
    )
    classify.run(snlim_peak=snlim_peak, key_classify=key_classify, remove_unnecessary=True)
    del classify
    return name_cube

def multirun_clfy(names_cube, num_cores=1):
    print('[Pipe_superprofile] BAYGAUD classify start')

    jobs = []
    for nm in names_cube:
        # everything the worker will need — no globals
        jobs.append({
            'name_cube': nm,
            'path_cube': dict_glob[nm]['path_cube'],
            'path_clfy': dict_glob[nm]['path_clfy'],
            'vdisp_low_intrinsic': dict_glob['vdisp_low_intrinsic'],
            'overwrite_classify': overwrite_classify,
        })

    if len(jobs) == 1:
        task_clfy(jobs[0])
    else:
        ctx = get_mp_ctx()  # spawn
        with ctx.Pool(processes=num_cores) as pool:
            with tqdm(total=len(jobs), leave=True) as pbar:
                for _ in pool.imap_unordered(task_clfy, jobs):
                    pbar.update()

    print('[Pipe_superprofile] BAYGAUD classify end')

# =========================
# Main fit task
# =========================
def task_main(job):
        
    # lazy imports (keeps workers lighter/safer)
    import astropy.units as u
    from shiftnstack import ShiftnStack, ShiftnStack_hermite

    name_cube         = job['name_cube']
    path_cube         = job['path_cube']
    path_mask         = job['path_mask']
    suffix            = job['suffix']
    mode              = job['mode']
    path_plot         = job['path_plot']
    path_temp         = job['path_temp']
    vdisp_low_intr    = job['vdisp_low_intrinsic']
    pbar_resample     = job['pbar_resample']
    nsample_resample  = job['nsample_resample']
    # dict_params       = job['dict_params']
    truth_from_resamp = job['truth_from_resampling']
    path_vf_secondary = job['path_vf_secondary'] if job['use_secondary_vf'] else None
    method_minimize   = job['method_minimize']

    if not os.path.exists(path_cube):
        return

    dict_config_stack = {
        'NGC1569':{
            'specmask':{
                'from':-20*(u.km/u.s),
                'to'  :20*(u.km/u.s),
                'mask':'inner',
                'mode':'velocity'
            },
            'correct_secondary_vf':False
        },
        'AGC224248':{
            'specmask':{
                'from':0,
                'to'  :32,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'AGC225847':{
            'specmask':{
                'from':60,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'AGC226178':{
            'specmask':{
                'from':25,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC130':{
            'specmask':{
                'from':0,
                'to'  :30,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC152':{
            # 'specmask':{
            #     'from':40,
            #     'to'  :120,
            #     'mask':'outer',
            #     'mode':'channel'
            # },
            'specmask':None,
            'correct_secondary_vf':False
        },
        'VCC169':{
            'specmask':None,
            'correct_secondary_vf':False
        },
        'VCC309':{
            'specmask':{
                'from':0,
                'to'  :140,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC328':{
            'specmask':{
                'from':20,
                'to'  :140,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC329':{
            'specmask':{
                'from':30,
                'to'  :200,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC331':{
            'specmask':{
                'from':20,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC340':{
            'specmask':{
                'from':0,
                'to'  :50,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC379':{
            'specmask':{
                'from':50,
                'to'  :230,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC381':{
            'specmask':{
                'from':20,
                'to'  :140,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC479':{
            'specmask':{
                'from':20,
                'to'  :110,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC566':{
            'specmask':{
                'from':10,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC613':{
            'specmask':{
                'from':20,
                'to'  :185,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC656':{
            'specmask':None,
            'correct_secondary_vf':False
        },
        'VCC667':{
            'specmask':None,
            'correct_secondary_vf':False
        },
        'VCC693':{
            'specmask':{
                'from':20,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC697':{
            'specmask':{
                'from':50,
                'to'  :400,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC699':{
            'specmask':{
                'from':20,
                'to'  :200,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1091':{
            'specmask':{
                'from':80,
                'to'  :250,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1098':{
            'specmask':{
                'from':60,
                'to'  :250,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1142':{
            'specmask':{
                'from':30,
                'to'  :200,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1411':{
            'specmask':{
                'from':0,
                'to'  :20,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1744':{
            'specmask':{
                'from':0,
                'to'  :20,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC1778':{
            'specmask':None,
            'correct_secondary_vf':False
        },
        'VCC1992':{
            'specmask':{
                'from':0,
                'to'  :20,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC2006':{
            'specmask':{
                'from':220,
                'to'  :230,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC2034':{
            'specmask':{
                'from':0,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCC2037':{
            'specmask':{
                'from':0,
                'to'  :20,
                'mask':'inner',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCCA055':{
            'specmask':{
                'from':30,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },        
        'VCCA069':{
            'specmask':{
                'from':20,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
        'VCCA070':{
            'specmask':{
                'from':20,
                'to'  :120,
                'mask':'outer',
                'mode':'channel'
            },
            'correct_secondary_vf':False
        },
    }

    if name_cube not in dict_config_stack:
        correct_vf_secondary = False
    else: 
        correct_vf_secondary = dict_config_stack[name_cube]['correct_secondary_vf']

    if mode == 'baygaud':
        path_clfy = job['path_clfy']
        if not os.path.exists(path_clfy):
            return
        if path_vf_secondary is not None and (not os.path.exists(path_vf_secondary)):
            path_vf_secondary = None
        sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask, path_vf_secondary=path_vf_secondary, correct_vf_secondary=correct_vf_secondary)
        sns.path_temp = path_temp
        sns.suffix    = suffix
    else:  # 'hermite'
        path_hermite = job['path_hermite']
        if not os.path.exists(path_hermite):
            return
        sns = ShiftnStack_hermite(path_cube, path_hermite, path_mask=path_mask, path_vf_secondary=path_vf_secondary, correct_vf_secondary=correct_vf_secondary)

    # if name_cube in dict_config_stack:
    #     if dict_config_stack[name_cube]['specmask'] is not None:
    #         sns.mask_specrange(dict_config_stack[name_cube]['specmask'])

    sns.run(stack_secondary=True)

    # if path_mask is not None and os.path.exists(path_mask):
    #     os.remove(path_mask)

    df_stacked = sns.df_stacked
    dict_stakd  = sns.dict_stacked
    df_stacked.to_string(path_cube.parent/'df_stacked.csv', index=False)

    from xgfit.xgfit import XGFIT
        
    xgfit = XGFIT(
        df_stacked, dict_stacked=dict_stakd, name_cube=name_cube,
        path_plot=path_plot, path_temp=path_temp,
        plot_autocorr=False, vdisp_low_intrinsic=vdisp_low_intr
    )
    del df_stacked, dict_stakd, sns
    gc.collect()
    

    if truth_from_resamp:
        xgfit.truth_from_resampling = True
    if fallback_to_2gfit:
        xgfit.fallback_to_2gfit = True    
    
    # xgfit.dict_params = dict_params
    
    xgfit.method_minimize = method_minimize
    xgfit.statistics      = statistics
    
    path_df_GFIT = job['path_temp'].parent/'info_stacked_GFIT.csv'
    
    # xgfit.guess_S32 = guess_S32
    
    if guess_from_whole:
        if suffix!='_' and path_df_GFIT.exists():
            df_GFIT = pd.read_csv(path_df_GFIT, sep=r'\s+')
            df_GFIT = df_GFIT.loc[(df_GFIT['Name']==job['name_cube']).__and__(df_GFIT['suffix']=='_')].reset_index(drop=True)
            if len(df_GFIT)>0:
                SNR2,SNR3 = df_GFIT.loc[0,['SNR2','SNR3']]
                if np.isfinite(SNR3):
                    F1,S1,F31,F32,F33,S31,S32,S33,B3 = df_GFIT.loc[0,['F1','S1','F31','F32','F33','S31','S32','S33','B3']]
                    xgfit.dict_preguess = {
                        # 'A31/A1':A31/A1,'A32/A1':A32/A1,'A33/A1':A33/A1,
                        'F31/F1':F31/F1,'F32/F1':F32/F1,'F33/F1':F33/F1,
                        'S31':S31,'S32':S32,'S33':S33
                    }
                    # xgfit.dict_prebound = {
                    #     # 'S31':[0.5*S31, 999],
                    #     'S32':[0.4*S32, 1.7*S32]
                    #     # 'S32':[0.65*S32,1.4*S32]
                    # }
                if np.isfinite(SNR2) and np.isnan(SNR3):
                    F1,S1,F21,F22,S21,S22 = df_GFIT.loc[0,['F1','S1','F21','F22','S21','S22']]
                    # Atot = A21+A22
                    xgfit.dict_preguess = {
                        'F31/F1':F21/F1,'F32/F1':F22/F1,'F33/F1':0.0001*F1,
                        'S31':S21,'S32':S22,'S33':S22+0.1
                    }
                    # xgfit.dict_prebound = {
                    #     # 'S31':[0.5*S31, 999],
                    #     # 'S32':[0.4*S22, 1.8*S22]
                    #     # 'S32':[0.65*S32,1.4*S32]
                    # }
            else:
                raise RuntimeError(f'Run whole first: {job['name_cube']}')

    xgfit.run(
        suffix=suffix,
        pbar_resample=pbar_resample,
        nsample_resample=nsample_resample
    )
    
    df        = xgfit.df.copy()
    df_params = xgfit.df_params.copy()

    df['suffix'] = suffix
    df_params['suffix'] = suffix

    # 컬럼 순서를 2번째 위치로 옮기고 싶다면:
    def move_suffix_to_second(df):
        cols = df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('suffix')))
        return df[cols]

    df        = move_suffix_to_second(df)
    df_params = move_suffix_to_second(df_params)

    df.to_string(        path_temp/f'{name_cube}{suffix}_plotfitter_temp_gfit.csv',  index=False)
    df_params.to_string( path_temp/f'{name_cube}{suffix}_plotfitter_temp_para.csv',  index=False)
    
    try: plt.close('all')
    except Exception: pass

    del xgfit,df,df_params
    gc.collect()
    return suffix


def multirun_main(num_threads=1):
    
    id_run  = dict_glob['id_run']
    tempdir = dict_glob['path_output']/f'temp_{id_run}'
    dict_glob['path_temp'] = tempdir
    if os.path.exists(tempdir): shutil.rmtree(tempdir)
    os.mkdir(tempdir)

    plotdir = dict_glob['path_output']/'png'
    dict_glob['path_plot'] = plotdir
    os.makedirs(plotdir, exist_ok=True)

    # decide pbar_resample first
    n_jobs_planned = len(dict_jobs)
    dict_glob['pbar_resample'] = (n_jobs_planned == 1)
    
    path_df_GFIT = dict_glob['path_output']/f'info_stacked_GFIT.csv'
    path_df_pram = dict_glob['path_output']/f'info_stacked_GFIT_params.csv'
    if path_df_GFIT.exists():
        df_GFIT = pd.read_csv(path_df_GFIT, sep=r'\s+')
    if path_df_pram.exists():
        df_pram = pd.read_csv(path_df_pram, sep=r'\s+')
    
    # build job list
    list_jobs = []
    for idx in sorted(dict_jobs.keys()):
        j = dict_jobs[idx].copy()
        if not bool_overwrite:
            name_cube,suffix = j['name_cube'],j['suffix']
            len_dfloc_GFIT = len_dfloc_pram = 0
            if path_df_GFIT.exists():
                dfloc_GFIT = df_GFIT[(df_GFIT['Name']==name_cube) & (df_GFIT['suffix']==suffix)]
                len_dfloc_GFIT = len(dfloc_GFIT)
            if path_df_pram.exists():
                dfloc_pram = df_pram[(df_pram['Name']==name_cube) & (df_pram['suffix']==suffix)]    
                len_dfloc_pram = len(dfloc_pram)
            if not len_dfloc_GFIT or not len_dfloc_pram: pass
            else: continue
            
            # if    np.isfinite(dfloc['N2'].item()) and np.abs(dfloc['B2'].item())>dfloc['N1'].item(): pass
            # if    np.isfinite(dfloc['N2'].item()) and dfloc['S22'].item()>dfloc['S1'].item()*5: pass
            # if np.isfinite(dfloc['N2'].item()): pass
            # if np.isfinite(dfloc['N3'].item()) and ((dfloc['A33'].item()/dfloc['N3'].item()<1) or ((dfloc['A32'].item()/dfloc['N3'].item()<1))): pass
            # else: continue
            
        j['mode']                 = dict_glob['mode']
        j['path_plot']            = dict_glob['path_plot']
        j['path_temp']            = dict_glob['path_temp']
        j['vdisp_low_intrinsic']  = dict_glob['vdisp_low_intrinsic']
        j['pbar_resample']        = dict_glob['pbar_resample']
        j['nsample_resample']     = dict_glob['nsample_resample']
        # j['dict_params']          = dict_glob['dict_params']
        j['truth_from_resampling']= bool(truth_from_resampling)
        j['use_secondary_vf']     = bool(use_secondary_vf)
        j['method_minimize']      = dict_glob['method_minimize']
        list_jobs.append(j)
        
    list_jobs.sort(key=lambda jj: (natsort_keygen()(jj['name_cube']), natsort_keygen()(jj['suffix'])))#, reverse=True)
        
    dict_done_cubes = {}
    dict_done_sufxs = {}
    names_cube = np.unique([j['name_cube'] for j in dict_jobs.values()])
    suffixes   = np.unique([j['suffix'] for j in dict_jobs.values()])
    for name in names_cube:
        dict_done_cubes[name] = dict([(sufx,False) for sufx in suffixes])
    for sufx in suffixes:
        dict_done_sufxs[sufx] = dict([(name,False) for name in names_cube])

    if not list_jobs:
        print('[Pipe_superprofile] No jobs to run.')
        return
    
    os.mkdir(dict_glob['path_temp']/'scanned')

    print('[Pipe_superprofile] GFIT start')
    print(f'       Path output: {dict_glob["path_output"]}')

    def periodic_message(interval, stop_event):
        while not stop_event.is_set():
            time.sleep(interval)
            try:
                texts_stat = natsorted(glob.glob(str(dict_glob['path_temp'] / 'stat*.txt')))
                results = []
                for fp in texts_stat:
                    with open(fp, 'r') as f:
                        splits = f.readline().strip().split()
                        results.append((splits[0], splits[1], splits[2], splits[3], splits[4]))
                if not results:
                    continue
                names_cube, suf, stats, iters, etas = zip(*results)
                df_stat = pd.DataFrame({
                    'Name': names_cube,
                    'Suffix': suf,
                    'Status': stats,
                    'Iteration': iters,
                    'ETA': etas
                }).sort_values(by=['Name','Suffix'], key=natsort_keygen())
                # path_df_stat = dict_glob['path_output']/f'Plotfit_stat_{dict_glob["id_run"]}.csv'
                path_df_stat = dict_glob['path_output']/f'Plotfit_stat.csv'
                with open(path_df_stat, "w") as f:
                    f.write(f"# {dict_glob['path_output'].name}\n")
                    f.write(f"#           num_threads = {num_threads}\n")
                    f.write(f"#      nsample_resample = {nsample_resample}\n")
                    f.write(f"#      use_secondary_vf = {bool(use_secondary_vf)}\n")
                    f.write(f"# truth_from_resampling = {bool(truth_from_resampling)}\n")
                    df_stat.to_string(f, index=False)
            except Exception as e:
                # print("[stat-writer]", e)
                continue


    def merge_temp_files(pattern: str, output_name: str):
        paths = sorted(glob.glob(str(dict_glob['path_temp'] / pattern)))
        if not paths:
            return

        out = dict_glob['path_output'] / output_name
        out_tmp = out.with_suffix('.tmp')  # <-- write here first

        # If output doesn't exist, "consume" the first temp file to create it
        start_idx = 0
        if not out.exists():
            # Prefer atomic move if same filesystem; if not, fallback to copy+remove
            try:
                os.replace(paths[0], out)  # atomic rename
            except OSError:
                shutil.copyfile(paths[0], out)
                try:
                    os.remove(paths[0])
                except FileNotFoundError:
                    pass
            start_idx = 1  # IMPORTANT: don't try to read the one we just consumed

        # Now merge any remaining temp files
        df_output = pd.read_csv(out, sep=r'\s+')
        for p in paths[start_idx:]:
            # Handle races: producer/another thread may have removed it
            if not os.path.exists(p):
                continue
            try:
                df_done = pd.read_csv(p, sep=r'\s+')
            except FileNotFoundError:
                continue

            df_output = pd.concat([df_output, df_done], ignore_index=True)
            df_output.drop_duplicates(subset=['Name', 'suffix'], keep='last', inplace=True)

            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        df_output.sort_values(by=['Name', 'suffix'], key=natsort_keygen(), inplace=True)
        df_output.to_string(out_tmp, index=False)
        os.replace(out_tmp, out)  # atomic on Linux

    def scan_and_merge(interval, stop_event):
        while not stop_event.is_set():
            time.sleep(interval)
            merge_temp_files('*_plotfitter_temp_gfit.csv',  'info_stacked_GFIT.csv')
            merge_temp_files('*_plotfitter_temp_para.csv',  'info_stacked_GFIT_params.csv')
            
                
    stop_event = Event()
    timer_1sec = Thread(target=periodic_message, args=(1, stop_event))
    timer_5sec = Thread(target=scan_and_merge,   args=(5, stop_event))
    timer_1sec.start()
    timer_5sec.start()
    # =========================================================

    if len(list_jobs) == 1:
        task_main(list_jobs[0])
    else:
        ctx = get_mp_ctx()
        with ctx.Pool(processes=num_threads, maxtasksperchild=1) as pool:
            with tqdm(total=len(list_jobs), leave=True, desc=dict_glob['suffix']) as pbar:
                for _ in pool.imap_unordered(task_main, list_jobs):
                    pbar.update()

    # stop stat-writer thread
    stop_event.set()
    timer_1sec.join()
    timer_5sec.join()

# =========================
# Mask workers (parallel)
# =========================
def worker_ring(multiplier, paths_cube, col_radius, path_df, ctx: MaskCtx):
    from run_makemask_ring import run_makemask as run_makemask_ring
    base = ctx.suffix
    suffix = f"{base}R{multiplier:.2f}{ctx.radtag}"
    masksuffix = f"_R{multiplier:.2f}{ctx.radtag}"
    out_csv = Path(ctx.path_output) / f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    run_makemask_ring(
        paths_cube,
        multiplier_radius_center=multiplier,
        width='beam',
        col_radius=col_radius,
        path_df=path_df
    )
    return {"kind": "ring", "suffix": suffix, "masksuffix": masksuffix}

def worker_angle(angle, angle_width, paths_cube, path_df, ctx: MaskCtx):
    from makemask_angle import makemask as makemask_angle
    base = ctx.suffix
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}"
    out_csv = Path(ctx.path_output) / f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df, savename=sv)
    return {"kind": "angle", "suffix": suffix}

def worker_angle_O05(angle, angle_width, paths_cube, path_df, ctx: MaskCtx):
    from makemask_angle import makemask as makemask_angle
    base = ctx.suffix
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}_O0.5r25"
    out_csv = Path(ctx.path_output) / f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df,
                       path_mask=p.parent/'mask/mask_O0.5r25.fits', savename=sv)
    return {"kind": "angle", "suffix": suffix}

def worker_angle_O10(angle, angle_width, paths_cube, path_df, ctx: MaskCtx):
    from makemask_angle import makemask as makemask_angle
    base = ctx.suffix
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}_O1.0r25"
    out_csv = Path(ctx.path_output) / f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df,
                       path_mask=p.parent/'mask/mask_O1.0r25.fits', savename=sv)
    return {"kind": "angle", "suffix": suffix}

from itertools import repeat


def dispatch_angle(task, paths_cube, path_df, ctx: MaskCtx):
    tag, a, w = task
    if tag == 'base':
        return worker_angle(a, w, paths_cube, path_df, ctx)
    elif tag == 'O05':
        return worker_angle_O05(a, w, paths_cube, path_df, ctx)
    elif tag == 'O10':
        return worker_angle_O10(a, w, paths_cube, path_df, ctx)
    return None

# =========================
# Main
# =========================

def main():
    
    from run_makemask_ellipse import run_makemask as run_makemask_ellipse

    timei = time.time()

    for homedir in homedirs:
        dict_jobs.clear()  # ← add this
        for k in [k for k in dict_glob if k not in (
        'mode','pbar_resample','nsample_resample','vdisp_low_intrinsic',
        'radtag','method_minimize','suffix','id_run','path_output',
        'path_plot','path_temp'
        )]:
            del dict_glob[k]
        homedir = Path(homedir)
        path_output = None
        
        # dict_glob['method_minimize'] = 'Nelder-Mead'
        path_output = homedir.parent/(homedir.name+suffix_path_output)
        
        # if dict_glob['mode']=='hermite':
        #     path_output = Path(str(path_output)+'_her')
        
        # path_output = homedir.parent/(homedir.name+'_stacktest_MOM')
        
        # path_output = homedir.parent/(homedir.name+'_3GFIT_LBFGSB')
        # path_output = homedir.parent/(homedir.name+'_3GFIT_Powell')
        
        # path_output = homedir.parent/(homedir.name+'_test')

        # discover cubes
        if nametype_galaxy is not None:
            paths_cube = [Path(p) for p in natsorted(glob.glob(str(homedir / f'{nametype_galaxy}/{nametype_cube}')))]
        if galaxies is not None:
            paths_cube = natsorted([Path(homedir/galaxy/nametype_cube) for galaxy in galaxies])
        names = []
        for i, path_cube in enumerate(paths_cube):
            wdir = path_cube.parent
            name_cube = wdir.name
            if dict_glob['mode']=='baygaud' and not os.path.exists(wdir/'segmts'):
                continue
            if dict_glob['mode']=='hermite' and not os.path.exists(wdir/'hermite.npy'):
                continue
            names.append(name_cube)
            dict_glob[name_cube] = {
                'path_cube': path_cube,
                'name_cube': name_cube,
                'path_mask': None,
                'path_vf_secondary': wdir/'cube_mom1.fits'
            }
            if dict_glob['mode']=='baygaud':
                dict_glob[name_cube]['path_clfy']   = wdir/f'segmts_merged_n_classified.{key_classify}'
            if dict_glob['mode']=='hermite':
                dict_glob[name_cube]['path_hermite'] = wdir/'hermite.npy'

        dict_glob['suffix'] = suffix
        dict_glob['id_run'] = str(time.time()).split(".")[-1]

        if path_output is None:
            path_output = homedir
        dict_glob['path_output'] = path_output
        os.makedirs(path_output, exist_ok=True)
        
        mask_ctx = MaskCtx(
            suffix=dict_glob['suffix'],
            radtag=dict_glob['radtag'],
            mode=dict_glob['mode'],
            path_output=str(dict_glob['path_output']),
        )

        # clean pre-existing
        if remove_temp:
            for d in glob.glob(str(path_output/'temp_*/')):
                shutil.rmtree(d, ignore_errors=True)
            for f in glob.glob(str(path_output/'Plotfit_stat*.csv')):
                try: os.remove(f)
                except: pass

        # if bool_overwrite:
        #     for f in glob.glob(str(path_output/'info_stacked*')):
        #         print('[Pipe_superprofile] Removing pre-existing fit data')
        #         try: os.remove(f)
        #         except: pass
        #     pngdir = path_output/'png'
        #     if os.path.exists(pngdir):
        #         print('[Pipe_superprofile] Removing pre-existing figure directory')
        #         shutil.rmtree(pngdir, ignore_errors=True)
        #         os.makedirs(pngdir, exist_ok=True)

        # classify
        if bool_do_clfy and dict_glob['mode']=='baygaud':
            multirun_clfy(names, num_cores=num_threads)

        # whole
        if bool_do_whole:
            dict_jobs.clear()

            cur_suffix = dict_glob['suffix']
            for name_cube in names:
                add_job(name_cube, cur_suffix, path_mask=None)


        # inner / outer (kept serial as original)
        if bool_do_inner:
            for m in multipliers:
                dict_glob['suffix'] = suffix + f'I{m}{dict_glob["radtag"]}'
                masksuffix = f'_I{m}{dict_glob["radtag"]}'
                run_makemask_ellipse(paths_cube, multiplier_radius=m, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
                out = path_output/f'info_stacked{dict_glob["suffix"]}_GFIT.csv'
                if out.exists():
                    continue
                for name_cube in names:
                    dict_glob[name_cube]['path_mask'] = dict_glob[name_cube]['path_cube'].parent / f'mask/mask{masksuffix}.fits'

        if bool_do_outer:
            for m in multipliers:
                dict_glob['suffix'] = suffix + f'O{m}{dict_glob["radtag"]}'
                masksuffix = f'_O{m}{dict_glob["radtag"]}'
                run_makemask_ellipse(paths_cube, multiplier_radius=m, col_radius=col_radius, path_df=path_data/'catalog/cat_diameters.csv')
                # (enqueue + multirun_main) block is commented in your original; left as-is.

        # =========================
        # NEW: Masks parallel, jobs serial enqueue
        # =========================
        # 1) rings
        from itertools import repeat

        from tqdm import tqdm

        job_specs = []
        if bool_do_rings and multipliers:
            with get_executor(num_threads) as ex:
                it = ex.map(
                    worker_ring,
                    multipliers,
                    repeat(paths_cube, len(multipliers)),
                    repeat(col_radius, len(multipliers)),
                    repeat(path_data/'catalog/cat_diameters.csv', len(multipliers)),
                    repeat(mask_ctx, len(multipliers)),   # 👈 pass ctx
                )
                res = list(tqdm(it, total=len(multipliers), desc="rings(mask)"))
            job_specs.extend([r for r in res if r])
            
        if bool_do_angles:
            # 2) common ellipse masks (serial; shared outputs)
            run_makemask_ellipse(paths_cube, multiplier_radius=0.5, col_radius='r25', path_df=path_data/'catalog/cat_diameters.csv')
            run_makemask_ellipse(paths_cube, multiplier_radius=1.0, col_radius='r25', path_df=path_data/'catalog/cat_diameters.csv')

            # 3) angles (parallel)
            # for W in (180, 90):
            for W in widths_angle:
                tasks = [('base', a, W) for a in PAs]

                # serial fallback
                if len(tasks) == 1:
                    res = [
                        dispatch_angle(task, paths_cube, path_data/'catalog/cat_diameters.csv', mask_ctx)
                        for task in tqdm(tasks, total=len(tasks), desc=f"angles(mask) W={W:03d}")
                    ]
                else:
                    with get_executor(num_threads) as ex:
                        it = ex.map(
                            dispatch_angle,
                            tasks,
                            repeat(paths_cube, len(tasks)),
                            repeat(path_data/'catalog/cat_diameters.csv', len(tasks)),
                            repeat(mask_ctx, len(tasks)),
                        )
                        res = list(tqdm(it, total=len(tasks), desc=f"angles(mask) W={W:03d}"))

                job_specs.extend([r for r in res if r])

            # 4) enqueue jobs in parent (serial)
            for spec in job_specs:
                if spec is None:
                    continue
                if spec["kind"] == "ring":
                    for name_cube in names:
                        path_mask = dict_glob[name_cube]['path_cube'].parent / f"mask/mask{spec['masksuffix']}.fits"
                        add_job(name_cube, spec['suffix'], path_mask)
                else:
                    for name_cube in names:
                        path_mask = dict_glob[name_cube]['path_cube'].parent / f"mask/mask{spec['suffix']}.fits"
                        add_job(name_cube, spec['suffix'], path_mask)

        multirun_main(num_threads=num_threads)

        # 6) cleanup
        try:
            shutil.rmtree(dict_glob['path_output']/f'temp_{dict_glob["id_run"]}')
            os.remove(dict_glob['path_output']/f'Plotfit_stat_{dict_glob["id_run"]}.csv')
        except Exception:
            pass

        print(f'[Pipe_superprofile] Finished for {path_output}')

    timef = time.time()
    print('[Pipe_superprofile] Done!')
    print('                    Total time elapsed: {}'.format(str(datetime.timedelta(seconds=timef-timei))))


if __name__ == "__main__":
    # 리눅스도 force=True 권장 (이미 설정돼 있어도 덮어씀)
    os.environ.setdefault('ASTROPY_SKIP_CONFIG_UPDATE', '1')
    # mp.set_start_method("spawn", force=True)
    mp.set_start_method("forkserver", force=True)
    main()