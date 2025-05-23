import glob
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from routines_baygaudpi import read_sgfit
from spectral_cube import SpectralCube
from tidier_colorbar import colorbar
import warnings
from spectral_cube.io.core import StokesWarning
warnings.simplefilter("ignore", StokesWarning)


class Filter:
    
    def __init__(self, path_cube, path_classified, path_mask=None):
        
        path_cube = Path(path_cube)
        path_classified = Path(path_classified)

        cube = SpectralCube.read(path_cube)
    
        self.data_cube = fits.getdata(path_cube)
        if(len(self.data_cube.shape)>3):
            self.data_cube=self.data_cube[0,:,:,:]
        self.data_cube = self.data_cube*(u.Jy/u.beam)
        self.spec_axis = cube.spectral_axis
        
        self.data_count = np.zeros((len(self.data_cube[0,:,0]), len(self.data_cube[0,0,:])), dtype=int)
        self.disps      = np.full(len(self.data_count.flatten()), np.nan)
        
        self.data_velo = fits.getdata(glob.glob(str(path_classified/'sgfit/*.1.fits'))[0])*(u.km/u.s)
        self.data_disp = fits.getdata(glob.glob(str(path_classified/'sgfit/*.2.fits'))[0])*(u.km/u.s)
        self.data_bkgr = fits.getdata(glob.glob(str(path_classified/'sgfit/*.3.fits'))[0])*(u.Jy/u.beam)
        self.data_nois = fits.getdata(glob.glob(str(path_classified/'sgfit/*.4.fits'))[0])*(u.Jy/u.beam)
        
        self.data_e_disp = fits.getdata(glob.glob(str(path_classified/'sgfit/*.2.e.fits'))[0])*(u.km/u.s)
        self.e_disps      = np.full(len(self.data_count.flatten()), np.nan)
        
        if len(self.data_velo.shape)>2:
            self.data_velo = self.data_velo[0,:,:]
            
        if path_mask is not None:
            data_mask = fits.getdata(path_mask)
            if len(data_mask.shape)>2:
                data_mask = data_mask[0,:,:]
            self.data_velo = np.where(np.isfinite(data_mask), self.data_velo, np.nan)
            
        self.path_cube = path_cube
    
    def run(self):
        
        data_disp = self.data_disp[None,:]
        data_velo = self.data_velo[None,:]
        data_nois = self.data_nois[None,:]
        data_bkgr = self.data_bkgr[None,:]
        data_e_disp = self.data_e_disp[None,:]
        
        mask = (self.spec_axis[:,None,None] > data_velo-3*data_disp)&\
               (self.spec_axis[:,None,None] < data_velo+3*data_disp)&\
               (self.data_cube > 2*data_nois + data_bkgr)&\
               (self.data_disp>1*self.data_e_disp)
               
        data_masked = np.where(mask, self.data_cube, np.nan)
        self.data_count = np.count_nonzero(~np.isnan(data_masked), axis=0)
        
        self.data_mask = np.where(self.data_count>=3, 1, np.nan)
        
        valid_disp_mask = self.data_count>=3
        self.disps[valid_disp_mask.flatten()] = data_disp[0,valid_disp_mask].value
        self.disps = self.disps[np.isfinite(self.disps)]
        
        self.e_disps[valid_disp_mask.flatten()] = data_e_disp[0,valid_disp_mask].value
        self.e_disps = self.e_disps[np.isfinite(self.e_disps)]
        
        dict_disp = {}
        for i, (disp, e_disp) in enumerate(zip(self.disps, self.e_disps)):
            dict_disp[i] = {'disp':disp,
                            'e_disp':e_disp}
        
        self.dict_disp = dict_disp
        
        # from pprint import pprint
        # pprint(self.dict_disp)


if __name__=='__main__':
    
    # wdir = '/home/mskim/workspace/research/data/spec_interp/10.3kms/THINGS/DDO63'

    # filter = Filter(      path_cube=wdir+'/cube.fits',
    #                 path_classified=wdir+'/segmts_merged_n_classified.3',
    #                 )
    # filter.run()


    # disps = filter.disps

    # import pylab as plt
    # fig, ax = plt.subplots()

    # bins = np.arange(0,100,1)
    # ax.hist(disps, bins=bins)

    # hist,be = np.histogram(disps, bins=bins)
    # bc = (be[:-1] + be[1:]) / 2

    # P_hist   = hist/np.sum(hist)
    # CDF_hist = np.cumsum(P_hist)

    # # self.hist_disp_bc[np.searchsorted(self.CDF_hist,0.999)]

    # plt.axvline(bc[np.searchsorted(CDF_hist,0.95)])
    # plt.axvline(bc[np.searchsorted(CDF_hist,0.99)])
    # plt.axvline(bc[np.searchsorted(CDF_hist,0.999)])

    # plt.savefig('yo1.png')
    
    # wdir = '/home/mskim/workspace/research/data/spec_interp/10.3kms/AVID/VCC1091'
    wdir = '/home/mandu/workspace/research/data/AVID_halfbeam/VCC656'

    
    data_disp   = fits.getdata(wdir+'/segmts_merged_n_classified.3/sgfit/sgfit.G3_1.2.fits')
    data_e_disp = fits.getdata(wdir+'/segmts_merged_n_classified.3/sgfit/sgfit.G3_1.2.e.fits')
    
    import pylab as plt
    fig, axs = plt.subplots(ncols=4, sharex=True, sharey=True)
    
    ax = axs[0]
    img = ax.imshow(data_e_disp/data_disp, clim=(0,3), cmap='jet', interpolation='none')
    colorbar(img)
    ax = axs[1]
    mask = np.where(data_e_disp<data_disp,1,0)
    ax.imshow(mask, cmap='Oranges', interpolation='none')
    ax = axs[2]
    mask = np.where(2*data_e_disp<data_disp,1,0)
    ax.imshow(mask, cmap='Oranges', interpolation='none')
    ax = axs[3]
    mask = np.where(3*data_e_disp<data_disp,1,0)
    ax.imshow(mask, cmap='Oranges', interpolation='none')

    ax.invert_yaxis()
    
    plt.show()
    # plt.savefig('yo.png')
    
    
    
    
    