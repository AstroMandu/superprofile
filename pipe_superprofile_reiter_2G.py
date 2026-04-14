from pathlib import Path
import os
from makemask_angle import makemask as makemask_angle
from plotfit import Plotfit
import pandas as pd
import pylab as plt
import matplotlib
import numpy as np
import shutil
import time
from natsort import natsorted, natsort_keygen
from threading import Event, Thread
from tqdm import tqdm
import multiprocessing as mp


matplotlib.use('Agg')




def task_main(g):
    
    if not g['path_cube'].exists():
        raise
    
    if g['stak_v2nd']:
        if 'path_v2nd' not in g: raise
        if g['path_v2nd'].exists()==False: raise
    else:
        g['path_v2nd'] = None
    
    if g['stackmode']=='baygaud':
        if   'path_clfy' not in g: raise
        if g['path_clfy'].exists()==False: raise
        from shiftnstack import ShiftnStack as SnS
        
    if g['stackmode']=='hermite':
        if   'path_her3' not in g: raise
        if g['path_her3'].exists()==False: raise
        g['path_clfy'] = g['path_her3']
        from shiftnstack import ShiftnStack_hermite as SnS
    
    path_df_gfit = g['path_outp']/'temp'/f'{g['name_job']}_plotfitter_temp_gfit.csv'
    path_df_para = g['path_outp']/'temp'/f'{g['name_job']}_plotfitter_temp_para.csv'
    # df_gfit_orig = pd.read_csv(path_df_gfit, sep='\\s+')
    
    if g['overwrite']==False:
        if path_df_gfit.exists()==False: pass
        # df_orig = pd.read_csv(path_df_gfit, sep='\\s+')
        df_orig = pd.read_csv(g['path_outp']/'info_stacked_GFIT.csv', sep='\\s+')
    
    suffixes = []
    for i,ii in enumerate(g['iterables']):
        if ii=='_': 
            suffixes.append(ii)
        else:
            suffixes.append(f'_A{ii:>03}_W{g['angle_width']:>03}')
    
    for i,ii in enumerate(g['iterables']):
        
        suffix = suffixes[i]

        if ii=='_': path_mask = g['path_mask']
        else:
            if g['itermode']=='angle':
                sv = g['path_cube'].parent / f"mask/mask{suffix}.fits"
                os.makedirs(sv.parent, exist_ok=True)
                makemask_angle(g['path_cube'],angle_center=float(ii),angle_width=g['angle_width'],savename=sv,
                               path_df=Path('/home/mskim/workspace/research/data/catalog/cat_diameters.csv'))
                path_mask = sv
                
        if g['overwrite']==False:
            if len(df_orig.loc[(df_orig['Name']==g['name_cube']).__and__(df_orig['suffix']==suffix)])>0:
                continue
        
        sns = SnS(g['path_cube'], g['path_clfy'], path_mask, path_vf_secondary=g['path_v2nd'])
        sns.run(stack_secondary=g['stak_v2nd'])
        
        df_stacked = sns.df_stacked
        dict_disp  = sns.dict_disp
        df_stacked.to_string(g['path_cube'].parent/'df_stacked.csv', index=False)
        
        plotfit = Plotfit(
            df_stacked, dict_disp=dict_disp, name_cube=g['name_cube'],path_plot=g['path_outp']/'png',path_temp=g['path_outp']/'temp',
            plot_autocorr=False,vdisp_low_intrinsic=0,
        )
        plotfit.method_minimize = 'Nelder-Mead'
        
        del df_stacked,dict_disp,sns
        
        if g['guess_from_prev']:
            if i==0: 
                if 'preguess' in g and g['preguess'] is not None:
                    A1,A21,A22,V21,V22,S21,S22,B2 = g['preguess']
                    plotfit.guess_V21=V21;plotfit.guess_V22=V22;
                    plotfit.guess_S21=S21;plotfit.guess_S22=S22;
                    plotfit.guess_A21=A21;plotfit.guess_A22=A22;
                    plotfit.guess_B2=B2
            else:
                suffix_prev = suffixes[i-1]
                df_GFIT = pd.read_csv(path_df_gfit,sep='\\s+')
                df_GFIT = df_GFIT.loc[(df_GFIT['Name']==g['name_cube']).__and__(df_GFIT['suffix']==suffix_prev)].reset_index(drop=True)
                if len(df_GFIT)!=1: 
                    print(g['name_cube'], suffix_prev)
                    print(df_GFIT)
                    raise
                params = df_GFIT.loc[0,['A1','A21','A22','V21','V22','S21','S22','B2']]
                if    params.isna().any(): pass
                else: 
                    A1,A21,A22,V21,V22,S21,S22,B2 = g['preguess']
                    plotfit.guess_V21=V21;plotfit.guess_V22=V22;
                    plotfit.guess_S21=S21;plotfit.guess_S22=S22;
                    plotfit.guess_A21=A21;plotfit.guess_A22=A22;
                    plotfit.guess_B2=B2
                
        plotfit.run(suffix=suffix,
                    pbar_resample=False,
                    # pbar_resample=True,
                    nsample_resample=g['nsample_resample'])
                
        plotfit.df = plotfit.df.copy()
        plotfit.df_params = plotfit.df_params.copy()
        
        plotfit.df['suffix']        = suffix
        plotfit.df_params['suffix'] = suffix
        
        def move_suffix_to_second(df):
            cols = df.columns.tolist()
            cols.insert(1, cols.pop(cols.index('suffix')))
            return df[cols]
        
        plotfit.df        = move_suffix_to_second(plotfit.df)
        plotfit.df_params = move_suffix_to_second(plotfit.df_params)
        
        if i==0:
            plotfit.df.to_string(       path_df_gfit,  index=False)
            plotfit.df_params.to_string(path_df_para,  index=False)
        else:
            df_gfit = pd.read_csv(path_df_gfit,sep='\\s+')
            df_gfit = pd.concat([df_gfit,plotfit.df], axis=0, ignore_index=True)
            df_gfit.to_string(path_df_gfit,index=False)
            df_para = pd.read_csv(path_df_para,sep='\\s+')
            df_para = pd.concat([df_para,plotfit.df_params], axis=0, ignore_index=True)
            df_para.to_string(path_df_para,index=False)
            
        try: plt.close('all')
        except Exception: pass
        
        del plotfit
    return suffix


def multirun_main(dict_multirun):

    def periodic_message(interval, stop_event):
        while not stop_event.is_set():
            time.sleep(interval)
            try:
                texts_stat = natsorted(list((dict_info['path_temp'].glob('stat*.txt'))))
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
                path_df_stat = dict_info['path_outp']/f'Plotfit_stat.csv'
                with open(path_df_stat, "w") as f:
                    f.write(f"# {dict_info['path_outp'].name}\n")
                    f.write(f"#           num_threads = {dict_info['num_threads']}\n")
                    # f.write(f"#      nsample_resample = {dict_info['nsample_resample']}\n")
                    # f.write(f"#      use_secondary_vf = {bool(use_secondary_vf)}\n")
                    # f.write(f"# truth_from_resampling = {bool(truth_from_resampling)}\n")
                    df_stat.to_string(f, index=False)
            except Exception as e:
                print("[stat-writer]", e)
                continue

    def scan_and_merge(interval, stop_event):
        while not stop_event.is_set():
            time.sleep(interval)

            paths_done = list(dict_info['path_temp'].glob('*_plotfitter_temp_gfit.csv'))
            if not len(paths_done): 
                continue
            
            path_output = dict_info['path_outp']/f'info_stacked_GFIT.csv'
            if not path_output.exists():
                shutil.copyfile(paths_done[0], path_output)
                continue
            
            df_output = pd.read_csv(path_output, sep=r'\s+')
            for path_done in paths_done:
                df_done   = pd.read_csv(path_done, sep=r'\s+')
                df_output = pd.concat([df_output, df_done], axis=0, ignore_index=True)
                df_output.drop_duplicates(subset=['Name','suffix'], keep='last', inplace=True)
            df_output.sort_values(by=['Name','suffix'],key=natsort_keygen(),inplace=True)
            df_output.to_string(path_output, index=False)
                
            paths_done = list(dict_info['path_temp'].glob('*_plotfitter_temp_para.csv'))
            if not len(paths_done): 
                continue
            
            path_output = dict_info['path_outp']/f'info_stacked_GFIT_params.csv'
            if not path_output.exists():
                shutil.copyfile(paths_done[0], path_output)
                continue
            
            df_output = pd.read_csv(path_output, sep=r'\s+')
            for path_done in paths_done:
                df_done   = pd.read_csv(path_done, sep=r'\s+')
                df_output = pd.concat([df_output, df_done], axis=0, ignore_index=True)
                df_output.drop_duplicates(subset=['Name','suffix'], keep='last', inplace=True)
            df_output.sort_values(by=['Name','suffix'],key=natsort_keygen(),inplace=True)
            df_output.to_string(path_output, index=False)
    
    dict_info = dict_multirun['dict_info']
    dict_jobs = dict_multirun['dict_jobs']
    
    if (dict_info['path_outp']/'temp').exists():
        shutil.rmtree(dict_info['path_outp']/'temp')
    
    # build job list
    list_jobs = []
    for idx in sorted(dict_jobs.keys()):
        j = dict_jobs[idx].copy()
        # if not bool_overwrite:
        #     name_cube = j['name_cube']
        #     len_dfloc_GFIT = len_dfloc_pram = 0
        #     if path_df_GFIT.exists():
        #         dfloc_GFIT = df_GFIT[(df_GFIT['Name']==name_cube) & (df_GFIT['suffix']==suffix)]
        #         len_dfloc_GFIT = len(dfloc_GFIT)
        #     if path_df_pram.exists():
        #         dfloc_pram = df_pram[(df_pram['Name']==name_cube) & (df_pram['suffix']==suffix)]    
        #         len_dfloc_pram = len(dfloc_pram)
        #     if not len_dfloc_GFIT or not len_dfloc_pram: pass
        #     else: continue

        list_jobs.append(j)
        
    list_jobs.sort(key=lambda jj: natsort_keygen()(jj['name_cube']))#, reverse=True)
        
    if not list_jobs:
        print('[Pipe_superprofile] No jobs to run.')
        return
    
    print('[Pipe_superprofile] GFIT start')
    print(f'       Path output: {dict_info["path_outp"]}')

    stop_event = Event()
    timer_1sec = Thread(target=periodic_message, args=(1, stop_event))
    timer_5sec = Thread(target=scan_and_merge,   args=(5, stop_event))
    timer_1sec.start()
    timer_5sec.start()
    # =========================================================

    if len(list_jobs) == 1:
        task_main(list_jobs[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=dict_info['num_threads']) as pool:
            with tqdm(total=len(list_jobs), leave=True) as pbar:
                for _ in pool.imap_unordered(task_main, list_jobs):
                    pbar.update()

    # stop stat-writer thread
    stop_event.set()
    timer_1sec.join()
    timer_5sec.join()
    
if __name__=='__main__':
    
    # dict_jobs['VCC566'] = {
    #     'stackmode':'baygaud',
    #     'name_cube':'VCC566',
    #     'path_cube':Path('/home/mskim/workspace/research/data/AVID_hann/VCC566/cube.fits'),
    #     'path_mask':Path('/home/mskim/workspace/research/data/AVID_hann/VCC566/cube_mom1.fits'),
    #     'path_clfy':Path('/home/mskim/workspace/research/data/AVID_hann/VCC566/segmts_merged_n_classified.3'),
    #     'path_outp':Path('/home/mskim/workspace/research/data/AVID_hann_3GFIT_test'),

    #     'stak_v2nd':True,    
    #     'path_v2nd':Path('/home/mskim/workspace/research/data/AVID_hann/VCC566/cube_mom1.fits'),
        
    #     'itermode'   :'angle',
    #     'angle_width':180,
    #     'iterables'  :['_']+[f'{ii:>03}' for ii in range(0,360,15)],
    #     'start_from' :'000',
        
    #     'nsample_resample': 599,
    #     'guess_from_prev':  True,
    #     'overwrite': True
    # }
    
    dict_dict_preguess = {
        'BAYMOM_AVID':{
            'AGC225847_W120':{'start_from':'_A090_W120','itereverse':0},
               'VCC130_W120':{'start_from':'_A135_W120','itereverse':0},
               'VCC152_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC169_W120':{'start_from':'_A270_W120','itereverse':0},
               'VCC309_W120':{'start_from':'_A225_W120','itereverse':0},
               'VCC322_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC328_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC329_W120':{'start_from':'_A225_W120','itereverse':0},
               'VCC331_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC334_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC340_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC379_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC381_W120':{'start_from':'_A270_W120','itereverse':0},
               'VCC566_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC613_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC656_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC667_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC693_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC697_W120':{'start_from':'_A000_W120','itereverse':0},
               'VCC699_W120':{'start_from':'_A000_W120','itereverse':0},
              'VCC1091_W120':{'start_from':'_A000_W120','itereverse':0},
              'VCC1411_W120':{'start_from':'_A000_W120','itereverse':0},
              'VCC1778_W120':{'start_from':'_A000_W120','itereverse':1},
              'VCC1992_W120':{'start_from':'_A000_W120','itereverse':0},
              'VCC2006_W120':{'start_from':'_A000_W120','itereverse':0},
              'VCC2034_W120':{'start_from':'_A225_W120','itereverse':0},
              'VCC2037_W120':{'start_from':'_A000_W120','itereverse':0},
        },
        'BAYMOM_LT':{
            # 'CVnIdwA_W120': {'start_from':'_A135_W120'},
            # 'DDO43_W120':   {'start_from':'_A000_W120'},
            # 'DDO46_W120':   {'start_from':'_A045_W120'},
            # 'DDO47_W120':   {'start_from':'_A090_W120'},
            # 'DDO50_W120':   {'start_from':'_A000_W120'},
            # 'DDO52_W120':   {'start_from':'_A270_W120'},
            # 'DDO53_W120':   {'start_from':'_A225_W120'},
            # 'DDO63_W120':   {'start_from':'_A000_W120'},
            # 'DDO69_W120':   {'start_from':'_A180_W120'},
            # 'DDO70_W120':   {'start_from':'_A000_W120'},
            # 'DDO75_W120':   {'start_from':'_A225_W120'},
            # 'DDO87_W120':   {'start_from':'_A000_W120'},
            # 'DDO101_W120':  {'start_from':'_A090_W120'},
            # 'DDO126_W120':  {'start_from':'_A000_W120'},
            # 'DDO133_W120':  {'start_from':'_A000_W120'},
            # 'DDO154_W120':  {'start_from':'_A000_W120'},
            # 'DDO155_W120':  {'start_from':'_A000_W120'},
            # 'DDO165_W120':  {'start_from':'_A000_W120'},
            'DDO167_W120':  {'start_from':'_A045_W120'},
            # 'DDO168_W120':  {'start_from':'_A045_W120'},
            # 'DDO187_W120':  {'start_from':'_A180_W120'},
            # 'DDO210_W120':  {'start_from':'_A180_W120'},
            'DDO216_W120':  {'start_from':'_A135_W120'},
            'F564-V3_W120': {'start_from':'_A180_W120'},
            # 'Haro29_W120':  {'start_from':'_A000_W120'},
            'Haro36_W120':  {'start_from':'_A180_W120'},
            'IC10_W120':    {'start_from':'_A270_W120'},
            'IC1613_W120':  {'start_from':'_A270_W120'},
            'LGS3_W120':    {'start_from':'_A180_W120'},
            'M81DWA_W120':  {'start_from':'_A060_W120'},
            'Mrk178_W120':  {'start_from':'_A240_W120'},
            'NGC1569_W120': {'start_from':'_A015_W120'},
            # 'NGC2366_W120': {'start_from':'_A000_W120'},
            'NGC3738_W120': {'start_from':'_A270_W120'},
            'NGC4163_W120': {'start_from':'_A270_W120'},
            # 'NGC4214_W120': {'start_from':'_A225_W120'},
            'SAGDIG_W120':  {'start_from':'_A045_W120'},
            # 'UGC8508_W120': {'start_from':'_A135_W120'},
            # 'VIIZw403_W120':{'start_from':'_A000_W120'},
            # 'WLM_W120':     {'start_from':'_A000_W120'},
        },
        'BAYMOM_Rory':{
            'snap00_W120':{'start_from':'_A225_W120','itereverse':False},
            'snap09_W120':{'start_from':'_A000_W120','itereverse':False},
            'snap10_W120':{'start_from':'_A180_W120','itereverse':False},
            'snap13_W120':{'start_from':'_A000_W120','itereverse':True},
        },
    }

    dict_jobs  = {}
    dict_info = {}

    stakmode='baygaud'
    # key_preguess = 'BAYMOM_LT'
    # path_output_source = Path('/home/mskim/workspace/research/data/LITTLE_THINGS_halfbeam_3GFIT_BAYMOM')
    # paths_cube =         Path('/home/mskim/workspace/research/data/LITTLE_THINGS_halfbeam').glob('*/cube.fits')
    
    key_preguess = 'BAYMOM_AVID'
    path_output_source = Path('/home/mskim/workspace/research/data/AVID_hann_2GFIT_BAYMOM')
    paths_cube =         Path('/home/mskim/workspace/research/data/AVID_hann').glob('*/cube.fits')
    
    # key_preguess = 'BAYMOM_Rory'
    # path_output_source = Path('/home/mskim/workspace/research/data/Rory/RPfiles_0.01mJy_3GFIT_BAYMOM')
    # paths_cube =         Path('/home/mskim/workspace/research/data/Rory/RPfiles_0.01mJy').glob('*/cube.fits')
    
    # key_preguess='W120_BAYMOM'; angle_width = 120
    # key_preguess='W180_BAYMOM'; angle_width = 180
    # path_output_source = Path('/home/mskim/workspace/research/data/AVID_hann_3GFIT_BAYMOM')
    
    dict_info['path_outp']=Path(str(path_output_source)+'_reiter')

    # stakmode='hermite'
    # key_preguess='HERMOM_alt'
    # # path_output_source = Path('/home/mskim/workspace/research/data/AVID_hann_3GFIT_HERMOM')
    # path_output_source = Path('/home/mskim/workspace/research/data/LITTLE_THINGS_halfbeam_3GFIT_HERMOM')
    # paths_cube =         Path('/home/mskim/workspace/research/data/LITTLE_THINGS_halfbeam').glob('*/cube.fits')
    # dict_info['path_outp']=Path(str(path_output_source)+'_reiter_alt')
    
    # stakmode='baygaud'
    # key_preguess='W120_BAYMOM_Rory'; angle_width = 120
    # path_output_source = Path('/home/mskim/workspace/research/data/Rory/RPfiles_0.1mJy_3GFIT_BAYMOM')
    # dict_info['path_outp']=Path(str(path_output_source)+'_reiter')
    # paths_cube = Path('/home/mskim/workspace/research/data/Rory/RPfiles_0.1mJy').glob('*13/cube.fits')


    dict_preguess = dict_dict_preguess[key_preguess]
    # paths_cube = Path('/home/mskim/workspace/research/data/AVID_hann/').glob('*/cube.fits')
    dict_info['num_threads'] = 64
    dict_info['path_temp']=dict_info['path_outp']/'temp'
    dict_info['path_plot']=dict_info['path_outp']/'png'
    
    # paths_cube = [
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC130/cube.fits'),
    #     Path('/home/mskim/workspace/research/data/AVID_hann/VCC328/cube.fits'),
    #     Path('/home/mskim/workspace/research/data/AVID_hann/VCC331/cube.fits'),
    #     Path('/home/mskim/workspace/research/data/AVID_hann/VCC334/cube.fits'),
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC566/cube.fits'),
    #     Path('/home/mskim/workspace/research/data/AVID_hann/VCC613/cube.fits'),
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC667/cube.fits'),
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC1091/cube.fits'),
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC1778/cube.fits'),
    #     # Path('/home/mskim/workspace/research/data/AVID_hann/VCC2006/cube.fits'),
    #     Path('/home/mskim/workspace/research/data/AVID_hann/VCC2034/cube.fits'),
    # ]
    
    for path_cube in paths_cube:
        galname = path_cube.parent.name
        
        # dict_jobs[galname+f'_{angle_width}'] = {
        #     'stackmode':stakmode,'name_cube':galname,
        #     'path_cube':Path(path_cube.parent/'cube.fits'),
        #     'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
        #     'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
        #     'path_her3':Path(path_cube.parent/'hermite.npy'),
        #     'path_outp':dict_info['path_outp'],

        #     'stak_v2nd':True,    
        #     'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
        #     'itermode'   :'angle',
        #     'angle_width':angle_width,
        #     'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
        #     'start_from' :None,
        #     'itereverse' :False,
            
        #     # 'nsample_resample': 5000,
        #     'nsample_resample': 101,
        #     'guess_from_prev':  True,
        #     'overwrite': True
        # }
        
        # dict_jobs[galname+'_W045'] = {
        #     'stackmode':stakmode,'name_cube':galname,
        #     'path_cube':Path(path_cube.parent/'cube.fits'),
        #     'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
        #     'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
        #     'path_her3':Path(path_cube.parent/'hermite.npy'),
        #     'path_outp':dict_info['path_outp'],

        #     'stak_v2nd':True,    
        #     'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
        #     'itermode'   :'angle',
        #     'angle_width':45,
        #     # 'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
        #     # 'start_from' :None,
        #     # 'itereverse' :False,
            
        #     # 'nsample_resample': 5000,
        #     'nsample_resample': 101,
        #     'guess_from_prev':  True,
        #     'overwrite': True
        # }
        # dict_jobs[galname+'_W060'] = {
        #     'stackmode':stakmode,'name_cube':galname,
        #     'path_cube':Path(path_cube.parent/'cube.fits'),
        #     'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
        #     'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
        #     'path_her3':Path(path_cube.parent/'hermite.npy'),
        #     'path_outp':dict_info['path_outp'],

        #     'stak_v2nd':True,    
        #     'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
        #     'itermode'   :'angle',
        #     'angle_width':60,
        #     # 'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
        #     # 'start_from' :None,
        #     # 'itereverse' :False,
            
        #     'nsample_resample': 5000,
        #     # 'nsample_resample': 101,
        #     'guess_from_prev':  True,
        #     'overwrite': True
        # }
        # dict_jobs[galname+'_W090'] = {
        #     'stackmode':stakmode,'name_cube':galname,
        #     'path_cube':Path(path_cube.parent/'cube.fits'),
        #     'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
        #     'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
        #     'path_her3':Path(path_cube.parent/'hermite.npy'),
        #     'path_outp':dict_info['path_outp'],

        #     'stak_v2nd':True,    
        #     'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
        #     'itermode'   :'angle',
        #     'angle_width':90,
        #     # 'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
        #     # 'start_from' :None,
        #     # 'itereverse' :False,
            
        #     # 'nsample_resample': 5000,
        #     'nsample_resample': 101,
        #     'guess_from_prev' : True,
        #     'overwrite': True
        # }
        dict_jobs[galname+'_W120'] = {
            'stackmode':stakmode,'name_cube':galname,
            'path_cube':Path(path_cube.parent/'cube.fits'),
            'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
            'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
            'path_her3':Path(path_cube.parent/'hermite.npy'),
            'path_outp':dict_info['path_outp'],

            'stak_v2nd':True,    
            'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
            'itermode'   :'angle',
            'angle_width':120,
            # 'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
            # 'start_from' :None,
            # 'itereverse' :False,
            
            # 'nsample_resample': 5000,
            'nsample_resample': 101,
            'guess_from_prev':  True,
            'overwrite': True
        }
        # dict_jobs[galname+'_W180'] = {
        #     'stackmode':stakmode,'name_cube':galname,
        #     'path_cube':Path(path_cube.parent/'cube.fits'),
        #     'path_mask':Path(path_cube.parent/'cube_mom1.fits'),
        #     'path_clfy':Path(path_cube.parent/'segmts_merged_n_classified.3'),
        #     'path_her3':Path(path_cube.parent/'hermite.npy'),
        #     'path_outp':dict_info['path_outp'],

        #     'stak_v2nd':True,    
        #     'path_v2nd':Path(path_cube.parent/'cube_mom1.fits'),
            
        #     'itermode'   :'angle',
        #     'angle_width':180,
        #     # 'iterables'  :[f'{ii:>03}' for ii in range(0,360,15)],
        #     # 'start_from' :None,
        #     # 'itereverse' :False,
            
        #     # 'nsample_resample': 5000,
        #     'nsample_resample': 101,
        #     'guess_from_prev':  True,
        #     'overwrite': True
        # }
    
    for key in list(dict_jobs.keys()):
        if key not in dict_preguess:
            del dict_jobs[key]
            
    import re
    df_source = pd.read_csv(path_output_source/'info_stacked_GFIT.csv',sep='\\s+')
    for key,val in dict_preguess.items():
        if key not in dict_jobs: continue
        
        PA_start_from = int((re.search(r"A(\d{3})", val['start_from'])).group(1))
        
        df_loc = df_source.loc[(df_source['Name']==key.split('_')[0]).__and__(df_source['suffix']==val['start_from'])].reset_index(drop=True)
        params = df_loc.loc[0,['A1','A21','A22','V21','V22','S21','S22','B2']]
        if params.isna().any():
            dict_jobs[key]['preguess'] = None
        else: 
            dict_jobs[key]['preguess'] = params
        
        dict_jobs[key+'_A'] = dict_jobs[key].copy()
        dict_jobs[key+'_B'] = dict_jobs[key].copy()
        del dict_jobs[key]
        
        dict_jobs[key+'_A']['name_job'] = key+'_A'
        dict_jobs[key+'_B']['name_job'] = key+'_B'
        
        iterA = np.array([ii for ii in range(0,180,15)])   + PA_start_from
        iterB = np.array([ii for ii in range(0,-180,-15)]) + PA_start_from
        iterB = iterB[1:]
        iterA[iterA>=360] -=360
        iterB[iterB<0]    +=360
        
        dict_jobs[key+'_A']['iterables'] = iterA
        dict_jobs[key+'_B']['iterables'] = iterB
    
    dict_multirun = {
        'dict_info':dict_info,
        'dict_jobs':dict_jobs
    }

    # task_main(dict_jobs['VCC566'])
    
    multirun_main(dict_multirun)
        
    
        
        
        
    