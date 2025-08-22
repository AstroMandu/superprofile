# gauss_hermite.py
# Kim, Minsu
# Sejong University, Korea
# 22.09.30

# Fits Hermite h3 polynomial on each spaxels and saves into fits files (amplitude, velocity) 
# 

# Note: multiprocessing doesn't seem to work well inside the class
# I already tried building this code using class but it did not work

import glob
import multiprocessing
import os
import shutil
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from natsort import natsorted
from scipy.optimize import minimize
from spectral_cube import SpectralCube
from tqdm import tqdm
from pathlib import Path
from numba import njit

warnings.filterwarnings('ignore', category=RuntimeWarning)

dict_glob = {}

@njit
def model(y, a, h3, Z):
    return a*np.exp(-0.5*np.square(y))*(1 + h3/np.sqrt(6)*(2*np.sqrt(2)*np.power(y,3.)-3*np.sqrt(2)*y)) + Z

@njit 
def calc_y(xx,b,c):
    return (xx-b)/c

@njit
def chisq_gauss2(yy,model,noise):
    return 0.5*np.sum(np.square((yy-model)/noise))
    
def cost(params):
    a,b,c,h3,Z = params
    y = calc_y(dict_glob['xx'],b,c)
    model_y = model(y,a,h3,Z)
    return chisq_gauss2(dict_glob['yy'],model_y,dict_glob['noise'])

def job(js):
    
    j=js

    df = pd.DataFrame()
    df['A']      = np.full(dict_glob['NAXIS1'], np.nan)
    df['B']      = np.nan
    df['C']      = np.nan
    df['h3']     = np.nan
    df['Z']      = np.nan
    df['v_peak'] = np.nan

    for i in range(dict_glob['NAXIS1']):
        
        velo = dict_glob['data_velo'][j,i]
        disp = dict_glob['data_disp'][j,i]
        Mflx = dict_glob['data_Mflx'][j,i]
        psnr = dict_glob['data_psnr'][j,i]
        
        if np.isfinite(velo)==False:          continue
        if disp<1.0:                          continue
        if psnr<2.0:                          continue
        if velo<dict_glob['spec_min']:        continue
        if velo>dict_glob['spec_max']:        continue
        if disp>dict_glob['bandwidth']/2.355: continue

        dict_glob['yy'] = np.float64(dict_glob['data_cube'][:,j,i])
        
        bounds = np.array([
            [0.1*Mflx,3.0*Mflx],                                              # A
            [dict_glob['spec_min'], dict_glob['spec_max']],
            [0.0,100000.0],
            [-1.0, 1.0],                                                                # h3
            [-Mflx, Mflx]                                                     # Z
        ])
        
        guess = np.array([Mflx,           # A
                velo,          # B
                disp,    # C
                0.0,                  # h3
                0.0])                 # Z
        
        names = ['A', 'B', 'C', 'h3', 'Z']
        
        for idx, (val, (low, high)) in enumerate(zip(guess, bounds)):
            if not (low <= val <= high):
                text = "Guess out of bounds\n"
                text += f'    (x,y)=({i},{j})\n'
                text += f'    {names[idx]}: [{bounds[idx][0]:.3e},{bounds[idx][1]:.3e}]    {guess[idx]:.3e}\n'
                
                print(text)
        
        res = minimize(cost, x0=guess, bounds=bounds, method='Nelder-Mead',tol=1e-100)
        # res = minimize(cost, x0=guess, bounds=bounds, method='L-BFGS-B')
                
        #EVALUATION
        if res.x[2]<dict_glob['disp_lower']:
            continue
        if res.x[0]/dict_glob['noise']<3:
            continue

        df.loc[i,'A']      = res.x[0]
        df.loc[i,'B']      = res.x[1]
        df.loc[i,'C']      = res.x[2]
        df.loc[i,'h3']     = res.x[3]
        df.loc[i,'Z']      = res.x[4]
        y = (res.x[1]-dict_glob['spec_axis'])/res.x[2]
        df.loc[i,'v_peak'] = dict_glob['spec_axis'][np.argmax(model(y,res.x[0],res.x[3],res.x[4]))]

    if(np.all(np.isnan(df['A']))==False):
        df.to_string(dict_glob['path_temp']/"{}.csv".format(j), index=False)
        return
    else:
        return
    
    
def makeplot(path_mom0, path_mom1):
    
    from astropy.wcs import WCS
    import pylab as plt
    
    wdir = os.path.dirname(path_mom0)
    galname = os.path.basename(os.path.dirname(path_mom0))

    data_mom0 = fits.getdata(path_mom0)
    data_mom1 = fits.getdata(path_mom1)
    
    hdr = fits.getheader(path_mom0)
    wcs = WCS(hdr).celestial

    fig, axs = plt.subplots(figsize=(7,3), ncols=2, sharex=True, sharey=True, subplot_kw={'projection':wcs}, tight_layout=True)
    
    axs[0].imshow(data_mom0, interpolation='none', clim=np.nanpercentile(np.where(data_mom0==0, np.nan, data_mom0), (0,95)), cmap='Blues_r')
    axs[1].imshow(data_mom1, interpolation='none', clim=np.nanpercentile(data_mom1, (5,95)), cmap='jet')
    # axs[0].invert_yaxis()
    
    # try:
    #     add_beam(axs[0], hdr, facecolor='none', edgecolor='white')
    # except KeyError:
    #     beam = Beam.from_fits_header(hdr)
    #     add_beam(axs[0], major=beam.major, minor=beam.minor, angle=beam.pa, facecolor='none', edgecolor='white')
    # add_scalebar(axs[0], length=1*u.kpc)
    
    # make_beammark(axs[0], scale_pix=(hdr['CDELT2']*u.deg), beam=Beam.from_fits_header(hdr))
    # make_scalemark(axs[0], scale_pix=(hdr['CDELT2']*u.deg), distance=10*u.Mpc)
    
    axs[0].text(0.05,0.95, f'{galname}', transform=axs[0].transAxes, color='white', va='top', ha='left')
    
    for ax in axs:
        ax.set_xlabel('RA (ICRS)')
        ax.set_ylabel('Dec. (ICRS)')
    
    
    # fig.suptitle(galname)

    # fig.savefig(wdir+"/moments.png")
    fig.savefig(wdir+'/hermite_{}.png'.format(galname))
    # plt.show()

    return 


def main(n_cores, path_cube, path_velo, path_disp, path_mask=None, vdisp_intrinsic=5.*(u.km/u.s)):

    # PARAMETERS:
    # - n_cores:    number of cores(threads) that will be used
    # - path_cube:  path to the cube
    # - path_vf:    path to the velocity refererence (used as initial guess in the fitting)
    # - path_vdisp: (optional) path to the velocity dispersion referrence (used as initial guess in the fitting)
    # - path_mask:  (optional) path to the mask
    # - multiplier_specax: multiplier to the spectral axis (ex. use 0.0001 to convert m/s to km/s)

    # RETURNS:
    # - Nothing
    # - Generates hermite_amp.fits and hermite_vel.fits in the directory where cube is

    # SETUP =============================================================================
    path_cube = Path(path_cube)
    wdir = path_cube.parent # Working directory
    name_cube = wdir.name
    
    hedr_cube = fits.getheader(path_cube) # Header, cube
    data_cube = fits.getdata(path_cube) # Data, cube
    if(len(data_cube.shape)>3):
        data_cube = np.squeeze(data_cube, axis=tuple(np.arange(len(data_cube.shape)-3)))
        
    noise = np.nanstd(np.hstack([data_cube[0,:,:],data_cube[-1,:,:]]).flatten())

    hedr_velo = fits.getheader(path_velo)
    data_velo = fits.getdata(path_velo) 
    if(len(data_velo.shape)>2):
        data_velo = np.squeeze(data_velo, axis=tuple(np.arange(len(data_velo.shape)-2)))

    if path_mask is not None:
        data_mask = fits.getdata(path_mask)
        if(len(data_mask.shape)>2):
            data_mask = np.squeeze(data_mask, axis=tuple(np.arange(len(data_mask.shape)-2)))

        data_velo[np.isnan(data_mask)] = np.nan

    data_disp = fits.getdata(path_disp) 
    if(len(data_disp.shape)>2):
        data_disp = np.squeeze(data_disp, axis=tuple(np.arange(len(data_disp.shape)-2)))

    spec_axis = SpectralCube.read(path_cube).with_spectral_unit(u.m/u.s, velocity_convention='optical').spectral_axis
    
    vdisp_channel   = np.abs(np.diff(spec_axis)[0])/2.355 * 2
    vdisp_lower     = np.sqrt((vdisp_intrinsic.to(u.m/u.s))**2 + vdisp_channel**2)

    # Generate a temporary directory that will store fit outputs
    path_temp = wdir/'temp_hermite'

    if(os.path.exists(path_temp)):
        shutil.rmtree(path_temp)
    os.mkdir(path_temp)

    # Make empty arrays those will contain fit outputs
    output       = np.full((6, len(data_velo[:,0]), len(data_velo[0,:])), np.nan)
    output_vpeak = np.full_like(data_velo, np.nan)
    
    dict_glob['NAXIS1']    = hedr_velo['NAXIS1']
    dict_glob['NAXIS2']    = hedr_velo['NAXIS2']
    
    dict_glob['data_cube'] = data_cube
    dict_glob['data_velo'] = data_velo
    dict_glob['data_disp'] = data_disp
    
    dict_glob['data_Mflx'] = np.max(data_cube, axis=(0))
    
    dict_glob['spec_axis'] = spec_axis.value
    dict_glob['spec_min']  = spec_axis.min().value
    dict_glob['spec_max']  = spec_axis.max().value
    dict_glob['bandwidth'] = (spec_axis.max() - spec_axis.min()).value
    
    dict_glob['disp_lower'] = vdisp_lower.value
    
    dict_glob['noise'] = noise
    dict_glob['data_psnr'] = dict_glob['data_Mflx']/noise
    
    dict_glob['xx'] = spec_axis.value
    dict_glob['path_temp'] = path_temp

    # END: SETUP =============================================================================

    # Count y-axis of the data
    lists = np.array(range(hedr_velo['NAXIS2']))

    print('[Hermite] Fitting in progress (n_cores = {})'.format(n_cores))

    # Setup multiprocessing
    pool = multiprocessing.Pool(processes=n_cores)

    # Distribute to cores the job, split using lists declared above
    with tqdm(total=len(lists)) as pbar:
        for _ in tqdm(pool.imap_unordered(job, lists), desc=name_cube):
            pbar.update()

    pool.close()
    pool.join()

    texts = natsorted(list(path_temp.glob('*.csv')))

    for text in texts:

        df = pd.read_csv(text, sep='\s+')
        j = int(os.path.splitext(os.path.basename(text))[0])
        
        output[0,j,:] = df['A']
        output[1,j,:] = df['B']
        output[2,j,:] = df['C']
        output[3,j,:] = df['h3']
        output[4,j,:] = df['Z']
        output[5,j,:] = df['v_peak']

        output_vpeak[j,:] = df['v_peak']

    # Modify the unit of the velocity output
    hedr_velo['BUNIT'] = 'm/s'
    # data_velo *= 1000.0

    # fits.writeto(wdir+'/hermite_vel.fits',  output[1,:,:]*1000.0, hedr_vf, overwrite=True)
    # fits.writeto(wdir+'/residual_vel.fits', output[1,:,:]*1000.0-data_velo, hedr_vf, overwrite=True)
    fits.writeto(wdir/'hermite_vel.fits',  output_vpeak, hedr_velo, overwrite=True)
    fits.writeto(wdir/'residual_vel.fits', output_vpeak-data_velo, hedr_velo, overwrite=True)

    # fits.writeto(wdir+'/hermite.fits', output, hedr_vf, overwrite=True)
    np.save(wdir/'hermite', output)
    shutil.rmtree(path_temp)
    
    makeplot(wdir/'cube_mom0.fits', wdir/'hermite_vel.fits')

    print('[Hermite] Outputs saved at {}'.format(wdir))

    
    # END OF THE LINE

if __name__=='__main__':
    
    import os
    from pathlib import Path

    from natsort import natsorted
    
    # paths_cube = [Path('/home/mskim/workspace/research/data/test/VCC1778/cube.fits')]
    # paths_cube = [Path('/home/mskim/workspace/research/data/AVID_halfbeam/VCC169/cube.fits')]
    
    paths_cube = Path('/home/mskim/workspace/research/data/AVID').glob('*/cube.fits')
    
    for path_cube in paths_cube:
    
        wdir = path_cube.parent
        galname = wdir.name
        
        if os.path.exists(wdir/'cube_mom1.fits')==False: 
            print(f'[Hermite] {galname} pass')
            continue
        
        print(wdir.name)
        main(50,
                wdir/'cube.fits',
                path_velo=wdir/'cube_mom1.fits',
                path_disp=wdir/'cube_mom2.fits',
                path_mask=wdir/'segmts_merged_n_classified.2/sgfit/sgfit.G3_1.1.fits',
                vdisp_intrinsic=0*(u.km/u.s))