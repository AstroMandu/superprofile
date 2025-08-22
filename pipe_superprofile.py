# -*- coding: utf-8 -*-
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
from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from plotfit import Plotfit
from run_makemask_ellipse import run_makemask as run_makemask_ellipse
from run_makemask_ring    import run_makemask as run_makemask_ring
from makemask_angle       import makemask as makemask_angle
from shiftnstack          import ShiftnStack, ShiftnStack_hermite

warnings.filterwarnings("ignore", category=RuntimeWarning)
matplotlib.use('Agg')

# =========================
# Config & Globals
# =========================
path_data = Path('/home/mskim/workspace/research/data')
homedirs = [
    
    # path_data/'THINGS_halfbeam'
    
    path_data/'AVID'
    
    # path_data/'test'
    
    # path_data/'AVID_halfbeam',
    # path_data/'test_clfy_2chan',
    # path_data/'test_clfy_3chan'
]

nametype_cube   = 'cube.fits'
nametype_galaxy = '*'

# multipliers = [round(x, 2) for x in np.arange(0.10, 1.50 + 0.001, 0.05)]
multipliers = [1.0]
# col_radius = 'r25';      radtag='r25'
col_radius = 'RHI(kpc)'; radtag='RHI'
suffix = '_'

snlim_peak   =  2
key_classify = '2'

use_secondary_vf = True
truth_from_resampling = 1
num_threads = 20

bool_overwrite     = 1
remove_temp        = 1

overwrite_classify = 0
bool_do_clfy       = 1

bool_do_whole  = 0
bool_do_inner  = 0
bool_do_outer  = 0

bool_do_rings  = 0

bool_do_angles = 1
angles = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]
# angles = [30]

bool_pack_output = 0
nsample_resample = 1499 # p~0.01

dict_glob = {}
dict_glob['dict_params']={
    'A21':'free',
    'A22':'free',
    'V21':'V1','V22':'V21',
    # 'V21':'free','V22':'free',
    'S21':'free',
    'S22':'free',
    'B2' :'free'
}
dict_glob['pbar_resample'] = False
dict_glob['nsample_resample'] = nsample_resample
dict_glob['mode'] = 'baygaud'
# dict_glob['mode'] = 'hermite'
dict_glob['vdisp_low_intrinsic'] = 0
dict_glob['radtag'] = radtag

dict_jobs = {}  # parent-only enqueue

# =========================
# Helpers
# =========================
def _base_suffix():
    base = dict_glob['suffix']
    return base

def add_job(name_cube, suffix, path_mask):
    index = len(dict_jobs)
    dict_jobs[index] = dict_glob[name_cube].copy()
    dict_jobs[index]['suffix']    = suffix
    dict_jobs[index]['path_mask'] = path_mask

# =========================
# Classify
# =========================
def task_clfy(name_cube):
    if (os.path.exists(dict_glob[name_cube]['path_clfy']) and not overwrite_classify):
        return
    from class_baypi_classify import Classify
    if not os.path.exists(dict_glob[name_cube]['path_cube'].parent / 'segmts'):
        return
    classify = Classify(
        dict_glob[name_cube]['path_cube'],
        path_baygaud=os.getenv('BAY'),
        vdisp_low_intrinsic=dict_glob['vdisp_low_intrinsic']
    )
    classify.run(snlim_peak=snlim_peak, key_classify=key_classify, remove_coolwarmhot=True)
    del classify

def multirun_clfy(names_cube, num_cores=1):
    print('[Pipe_superprofile] BAYGAUD classify start')
    lists = np.array(names_cube, dtype=object)
    if len(lists)==1:
        task_clfy(lists[0])
    else:
        pool = multiprocessing.Pool(processes=num_cores)
        with tqdm(total=len(lists), leave=True) as pbar:
            for _ in pool.imap_unordered(task_clfy, lists):
                pbar.update()
        pool.close(); pool.join()
    print('[Pipe_superprofile] BAYGAUD classify end')

# =========================
# Output writer
# =========================
def write_output(suffix):
    tempdir = dict_glob['path_temp']
    savename_gfit = dict_glob['path_output']/f'info_stacked{suffix}_GFIT.csv'
    savename_para = dict_glob['path_output']/f'info_stacked{suffix}_GFIT_params.csv'

    # gfit
    temps = natsorted(np.array(glob.glob(str(tempdir/f"*{suffix}_plotfitter_temp_gfit.txt"))))
    df = pd.concat([pd.read_csv(t, sep=r'\s+') for t in temps], ignore_index=True) if len(temps) else pd.DataFrame()
    if not df.empty:
        for c in df.columns:
            if c != 'Name':
                df[c] = pd.to_numeric(df[c])
        pd.options.display.float_format = '{:.3E}'.format
        df.to_string(savename_gfit, index=False)
        for t in temps: os.remove(t)

    # params
    temps = natsorted(np.array(glob.glob(str(tempdir/f"*{suffix}_plotfitter_temp_para.txt"))))
    dfp = pd.concat([pd.read_csv(t, sep=r'\s+') for t in temps], ignore_index=True) if len(temps) else pd.DataFrame()
    if not dfp.empty:
        for c in dfp.columns:
            if c not in ('Name','Reliable'):
                dfp[c] = pd.to_numeric(dfp[c])
        pd.options.display.float_format = '{:.3E}'.format
        dfp.to_string(savename_para, index=False)
        for t in temps: os.remove(t)

    print(f'[Pipe_superprofile] {suffix} completed:')
    if not dfp.empty: print(dfp)

def check_completed_suffix(suffix):
    tempdir = dict_glob['path_temp']
    tmps = tempdir.glob(f'*{suffix}_plotfitter_temp_para.txt')
    len_tmps = len(list(tmps))
    len_suffix = sum(job["suffix"] == suffix for job in dict_jobs.values())
        
    if len_tmps==len_suffix:
        write_output(suffix)
    return

# =========================
# Main fit task
# =========================
def task_main(index):
    name_cube = dict_jobs[index]['name_cube']
    path_cube = dict_jobs[index]['path_cube']
    path_mask = dict_jobs[index]['path_mask']
    suffix    = dict_jobs[index]['suffix']
    mode      = dict_glob['mode']
    
    path_plot = dict_glob['path_plot']
    path_temp = dict_glob['path_temp']

    path_vf_secondary = dict_jobs[index]['path_vf_secondary'] if use_secondary_vf else None


    if not os.path.exists(path_cube):
        return

    if mode=='baygaud':
        path_clfy = dict_jobs[index]['path_clfy']
        if not os.path.exists(path_clfy):
            return
        if path_vf_secondary is not None and (not os.path.exists(path_vf_secondary)):
            path_vf_secondary = None
        sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask, path_vf_secondary=path_vf_secondary)

    elif mode=='hermite':
        path_hermite = dict_jobs[index]['path_hermite']
        if not os.path.exists(path_hermite):
            return
        sns = ShiftnStack_hermite(path_cube, path_hermite, path_mask=path_mask, path_vf_secondary=path_vf_secondary)

    if name_cube == 'NGC1569':
        sns.mask_velrange(-20*(u.km/u.s), 20*(u.km/u.s))

    sns.run(stack_secondary=True)

    if path_mask is not None and os.path.exists(path_mask):
        os.remove(path_mask)

    df_stacked = sns.df_stacked
    dict_disp  = sns.dict_disp
    df_stacked.to_string(path_cube.parent/'df_stacked.csv', index=False)

    plotfit = Plotfit(
        df_stacked, dict_disp=dict_disp, name_cube=name_cube,
        path_plot=path_plot, path_temp=path_temp,
        plot_autocorr=False, vdisp_low_intrinsic=dict_glob['vdisp_low_intrinsic']
    )
    if truth_from_resampling:
        plotfit.truth_from_resampling = True
    plotfit.dict_params = dict_glob['dict_params']

    plotfit.run(
        suffix=suffix, 
        pbar_resample=dict_glob['pbar_resample'],
        nsample_resample=dict_glob['nsample_resample']
    )

    (dict_glob['path_temp']/f'{name_cube}{suffix}_plotfitter_temp_gfit.txt').write_text(plotfit.df.to_string(index=False))
    (dict_glob['path_temp']/f'{name_cube}{suffix}_plotfitter_temp_para.txt').write_text(plotfit.df_params.to_string(index=False))
    del plotfit

    check_completed_suffix(suffix)

def multirun_main(num_threads=1):
    id_run = dict_glob['id_run']
    tempdir = dict_glob['path_output']/f'temp_{id_run}'
    dict_glob['path_temp'] = tempdir
    if os.path.exists(tempdir): shutil.rmtree(tempdir)
    os.mkdir(tempdir)

    plotdir = dict_glob['path_output']/'png'
    dict_glob['path_plot'] = plotdir
    os.makedirs(plotdir, exist_ok=True)

    n_jobs = len(dict_jobs)
    if n_jobs==1:
        dict_glob['pbar_resample']=True

    print('[Pipe_superprofile] GFIT start')
    print(f'       Path output: {dict_glob["path_output"]}')
    maxtaskperchild = max(int(num_threads/max(n_jobs,1)), 1)
    print(f'maxtaskperchild: {maxtaskperchild}')
    pprint(dict_glob['dict_params'])

    lists = np.array(list(dict_jobs.keys()), dtype=object)

    interval = 1
    stop_event = Event()
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
                names_cube, suf, stats, iters, etas = zip(*results) if results else ([],[],[],[],[])
                if not names_cube:
                    continue
                df_stat = pd.DataFrame({
                    'Name': names_cube, 'Suffix': suf, 'Status': stats, 'Iteration': iters, 'ETA': etas
                }).sort_values(by=['Suffix','Name'])
                path_df_stat = dict_glob['path_output']/f'Plotfit_stat_{dict_glob["id_run"]}.csv'
                with open(path_df_stat, "w") as f:
                    f.write(f"# {dict_glob['suffix']}\n")
                    f.write(f"#           num_threads = {num_threads}\n")
                    f.write(f"#      nsample_resample = {nsample_resample}\n")
                    f.write(f"#      use_secondary_vf = {bool(use_secondary_vf)}\n")
                    f.write(f"# truth_from_resampling = {bool(truth_from_resampling)}\n")
                    df_stat.to_string(f, index=False)
            except Exception as e:
                print(e)
                continue

    timer_thread = Thread(target=periodic_message, args=(interval, stop_event))
    timer_thread.start()

    if len(lists)==1:
        task_main(lists[0])
    else:
        pool = multiprocessing.Pool(processes=num_threads, maxtasksperchild=maxtaskperchild)
        with tqdm(total=len(lists), leave=True, desc=dict_glob['suffix']) as pbar:
            for _ in pool.imap_unordered(task_main, lists):
                pbar.update()
        pool.close(); pool.join()

    stop_event.set()
    timer_thread.join()

# =========================
# Mask workers (parallel)
# =========================
def worker_ring(multiplier, paths_cube, col_radius, path_df):
    base = _base_suffix()
    suffix = f"{base}R{multiplier:.2f}{dict_glob['radtag']}"
    if dict_glob['mode']=='hermite': suffix+='_her'
    masksuffix = f"_R{multiplier:.2f}{dict_glob['radtag']}"
    out_csv = dict_glob['path_output']/f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    run_makemask_ring(
        paths_cube,
        multiplier_radius_center=multiplier,
        width='beam',
        col_radius=col_radius,
        path_df=path_df
    )
    return {"kind":"ring", "suffix":suffix, "masksuffix":masksuffix}

def worker_angle(angle, angle_width, paths_cube, path_df):
    base   = _base_suffix()
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}"
    if dict_glob['mode']=='hermite': suffix+='_her'
    out_csv = dict_glob['path_output']/f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df, savename=sv)
    return {"kind":"angle", "suffix":suffix}

def worker_angle_O05(angle, angle_width, paths_cube, path_df):
    base   = _base_suffix()
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}_O0.5r25"
    if dict_glob['mode']=='hermite': suffix+='_her'
    out_csv = dict_glob['path_output']/f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df,
                       path_mask=p.parent/'mask/mask_O0.5r25.fits', savename=sv)
    return {"kind":"angle", "suffix":suffix}

def worker_angle_O10(angle, angle_width, paths_cube, path_df):
    base   = _base_suffix()
    suffix = f"{base}A{angle:0>3}_W{angle_width:0>3}_O1.0r25"  # fix typo
    if dict_glob['mode']=='hermite': suffix+='_her'
    out_csv = dict_glob['path_output']/f"info_stacked{suffix}_GFIT.csv"
    if out_csv.exists():
        return None
    for p in paths_cube:
        sv = p.parent / f"mask/mask{suffix}.fits"
        os.makedirs(sv.parent, exist_ok=True)
        makemask_angle(p, angle_center=angle, angle_width=angle_width, path_df=path_df,
                       path_mask=p.parent/'mask/mask_O1.0r25.fits', savename=sv)
    return {"kind":"angle", "suffix":suffix}

# =========================
# Main
# =========================
timei = time.time()

for homedir in homedirs:
    homedir = Path(homedir)
    path_output = None
    path_output = homedir.parent/(homedir.name+'_2GFIT_G2n~G2b')

    # discover cubes
    paths_cube = [Path(p) for p in glob.glob(str(homedir / f'{nametype_galaxy}/{nametype_cube}'))]
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

    # clean pre-existing
    if remove_temp:
        for d in glob.glob(str(path_output/'temp_*/')):
            shutil.rmtree(d, ignore_errors=True)
        for f in glob.glob(str(path_output/'Plotfit_stat*.csv')):
            try: os.remove(f)
            except: pass

    if bool_overwrite:
        for f in glob.glob(str(path_output/'info_stacked*')):
            print('[Pipe_superprofile] Removing pre-existing fit data')
            try: os.remove(f)
            except: pass
        pngdir = path_output/'png'
        if os.path.exists(pngdir):
            print('[Pipe_superprofile] Removing pre-existing figure directory')
            shutil.rmtree(pngdir, ignore_errors=True)
            os.makedirs(pngdir, exist_ok=True)

    # classify
    if bool_do_clfy and dict_glob['mode']=='baygaud':
        multirun_clfy(names, num_cores=num_threads)

    # whole
    if bool_do_whole:
        out_csv = path_output / f'info_stacked_{dict_glob["suffix"]}GFIT.csv'
        if not out_csv.exists():
            multirun_main(num_threads=num_threads)

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
            multirun_main(num_threads=num_threads)

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
    job_specs = []
    if bool_do_rings and multipliers:
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            res = list(tqdm(
                ex.map(worker_ring, multipliers, 
                       [paths_cube]*len(multipliers),
                       [col_radius]*len(multipliers),
                       [path_data/'catalog/cat_diameters.csv']*len(multipliers)),
                total=len(multipliers), desc="rings(mask)"
            ))
        job_specs.extend([r for r in res if r])

    # 2) common ellipse masks (serial; shared outputs)
    run_makemask_ellipse(paths_cube, multiplier_radius=0.5, col_radius='r25', path_df=path_data/'catalog/cat_diameters.csv')
    run_makemask_ellipse(paths_cube, multiplier_radius=1.0, col_radius='r25', path_df=path_data/'catalog/cat_diameters.csv')

    # 3) angles (parallel)
    if bool_do_angles and angles:
        def _dispatch(tag, a, w):
            if tag=='base': return worker_angle(a, w, paths_cube, path_data/'catalog/cat_diameters.csv')
            # if tag=='O05':  return worker_angle_O05(a, w, paths_cube, path_data/'catalog/cat_diameters.csv')
            # if tag=='O10':  return worker_angle_O10(a, w, paths_cube, path_data/'catalog/cat_diameters.csv')

        tasks_180 = [('base', a, 180) for a in angles] + [('O05', a, 180) for a in angles] + [('O10', a, 180) for a in angles]
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            res = list(tqdm((ex.submit(_dispatch, k, a, w) for (k,a,w) in tasks_180),
                            total=len(tasks_180), desc="angles(mask) W=180"))
        job_specs.extend([f.result() for f in res if f.result()])
        
        # tasks_090 = [('base', a,  90) for a in angles] + [('O05', a,  90) for a in angles] + [('O10', a,  90) for a in angles]
        # with ProcessPoolExecutor(max_workers=num_threads) as ex:
        #     res = list(tqdm((ex.submit(_dispatch, k, a, w) for (k,a,w) in tasks_090),
        #                     total=len(tasks_090), desc="angles(mask) W=090"))
        # job_specs.extend([f.result() for f in res if f.result()])

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

    # 5) run fit workers
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
