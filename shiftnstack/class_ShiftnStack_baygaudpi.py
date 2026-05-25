import glob
import logging
import os
# from .class_Filter_genuine import Filter
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from natsort import natsorted
from radio_beam.beam import Beam
from routines_baygaudpi import gauss, read_ngfits
from run_makemask_ellipse import run_makemask as run_makemask_ellipse
from spectral_cube import SpectralCube
from spectral_cube.io.core import StokesWarning
from tqdm import tqdm

from .subroutines_shiftnstack import (argfind_nearest, shifter_gipsy,
                                      shifter_roll)

# matplotlib.use('Agg')

warnings.simplefilter("ignore", StokesWarning)


class ShiftnStack:
    
    def __init__(self, path_cube, path_classified, path_mask=None, bgsub=True, path_vf_secondary=None, correct_vf_secondary=False, 
                 secondary_mode:Literal['MOM','HER','HERMOM']='MOM'):
        # secondary_mode: 'moment' -> path_vf_secondary is a FITS moment-1 map (existing behaviour)
        #                 'hermite' -> path_vf_secondary is a hermite.npy; vf from [5] and disp from [2]
        path_cube       = Path(path_cube)
        path_classified = Path(path_classified) if path_classified is not None else None
        
        self.path_cube = path_cube
        self.path_clfy = path_classified
        self.path_mask = path_mask
        self.secondary_mode = secondary_mode
        
        self.name_cube = path_cube.parent.name
        
        hedr_cube = fits.getheader(path_cube)
        data_cube = fits.getdata(path_cube) #* (u.Jy/u.beam)
        if(len(data_cube.shape)>3): data_cube = data_cube[0,:,:,:]
        
        spec_axis = (SpectralCube.read(path_cube)).with_spectral_unit(u.km/u.s, velocity_convention='optical').spectral_axis.value
        chansep = np.diff(spec_axis)[0]
        
        self.hedr_cube = hedr_cube
        self.data_cube = data_cube 
        
        _shape2d = (hedr_cube['NAXIS2'], hedr_cube['NAXIS1'])

        hermite_vf_finite = None
        if path_vf_secondary is not None:
            path_vf_secondary = Path(path_vf_secondary)
            if secondary_mode in ('HER', 'HERMOM'):
                output_hermite = np.load(path_vf_secondary)
                data_vf_secondary   = (output_hermite[5,:,:] * u.m/u.s).to(u.km/u.s).value
                data_disp_secondary = (output_hermite[2,:,:] * u.m/u.s).to(u.km/u.s).value

                if secondary_mode == 'HERMOM':
                    hermite_vf_finite = np.isfinite(data_vf_secondary)  # before mom fill
                    path_mom1 = self.path_cube.parent / 'cube_mom1.fits'
                    path_mom2 = self.path_cube.parent / 'cube_mom2.fits'
                    if path_mom1.exists():
                        data_vf_mom = (fits.getdata(path_mom1) * (u.m/u.s)).to(u.km/u.s).value
                        if data_vf_mom.ndim > 2: data_vf_mom = data_vf_mom[0]
                        data_vf_secondary[~np.isfinite(data_vf_secondary)] = data_vf_mom[~np.isfinite(data_vf_secondary)]
                    if path_mom2.exists():
                        data_disp_mom = fits.getdata(path_mom2) / 1000.
                        data_disp_secondary[~np.isfinite(data_disp_secondary)] = data_disp_mom[~np.isfinite(data_disp_secondary)]

            elif secondary_mode == 'MOM':
                data_vf_secondary = (fits.getdata(path_vf_secondary) * (u.m/u.s)).to(u.km/u.s).value
                if data_vf_secondary.ndim > 2: data_vf_secondary = data_vf_secondary[0]
                if correct_vf_secondary:
                    import astropy.constants as const
                    c = const.c.to('km/s').value
                    data_vf_secondary = c * (1 / (1 - data_vf_secondary / c) - 1)
                data_disp_secondary = None
            else:
                raise ValueError(f"Unknown secondary_mode: '{secondary_mode}'")
            
        else:
            data_vf_secondary   = np.full(_shape2d, np.nan)
            data_disp_secondary = None

        self.data_vf_secondary   = data_vf_secondary
        self.data_disp_secondary = data_disp_secondary
        self._hermite_vf_finite  = hermite_vf_finite

        if(path_mask is not None):
            path_mask = Path(path_mask)
            data_mask = fits.getdata(path_mask)
            if(len(data_mask.shape)>2):
                data_mask = data_mask[0,:,:]
            # data_cube = np.where(np.isfinite(np.broadcast_to(data_mask, data_cube.shape)), data_cube, np.nan)
        else:
            data_mask = np.ones((hedr_cube['NAXIS2'],hedr_cube['NAXIS1']))
        self.data_mask = data_mask
        
        n = data_cube.shape[0]
        n_edge = max(n // 10, 10)  # at least 10 channels each side

        edge_channels = np.concatenate([
            data_cube[:n_edge, :, :],
            data_cube[-n_edge:, :, :]
        ], axis=0)  # shape (2*n_edge, ny, nx)

        std_channel = mad_std(edge_channels, ignore_nan=True)
        self.std_channel = std_channel * u.Jy/u.beam
           
        self.nopt     = 0
        self.len_nopt = 0
        if path_classified is not None and Path(path_classified).exists():
            nopt = len(glob.glob(str(path_classified/"ngfit/*G*_*.0.fits")))
            ngau_files = glob.glob(str(path_classified/"sgfit/*.7.fits"))
            if ngau_files:
                map_ngau = fits.getdata(ngau_files[0])
                self.len_nopt = int(np.nansum(map_ngau))
            if bgsub:
                bkgr_files = glob.glob(str(path_classified/"ngfit/*_1.3.fits"))
                if bkgr_files:
                    map_bkgr = fits.getdata(bkgr_files[0])
                    self.data_cube -= np.where(np.isfinite(map_bkgr), map_bkgr, 0)
            self.nopt = nopt
        
        self.spec_axis = spec_axis
        self.chansep    = chansep
        self.abschansep = np.abs(chansep)
        
        beam = Beam.from_fits_header(hedr_cube)
        self.beam = beam
        self.bM = beam.major.to(u.arcsec)
        self.bm = beam.minor.to(u.arcsec)
        self.cd = (hedr_cube['CDELT2']*u.degree).to(u.arcsec)
        self.len_sa = len(self.spec_axis)
        
        self.area_beam = self.beam.sr
        self.area_pixl = (self.cd**2).to(u.sr)
        self.pix_per_beam = self.area_beam/self.area_pixl
        
        self.dict_stacked = {}
        self.map_stack_method = np.zeros(_shape2d, dtype=np.int8)  # 0=none,1=baygaud,2=hermite,3=moment

        self.pbar   = False
        self.bgsub  = bgsub
        self.stat   = 'STBY'
        self.suffix = ''
        self.path_temp = None

        # Logging
        self.logger = None
        self._log_pixel_records = []  # list of (x, y, method_label, n_components)
        
    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def setup_logger(self, suffix: str = '', log_level: int = logging.INFO) -> None:
        """Create log/<name_cube><suffix>_<timestamp>.log under the cube's parent directory.

        Parameters
        ----------
        suffix : str
            Run suffix (e.g. '_I0.5r25'). Stored on self so writestat can use it.
        log_level : int
            Logging level for the file handler. Use logging.DEBUG to get
            per-pixel lines; logging.INFO (default) skips them with no
            formatting overhead.
        """
        self.suffix = suffix
        self._log_pixel_records = []

        log_dir = self.path_cube.parent / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name  = f"{self.name_cube}{suffix}_{timestamp}.log"
        log_path  = log_dir / log_name

        logger = logging.getLogger(f"sns.{self.name_cube}{suffix}.{timestamp}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.handlers:
            logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)-8s  %(message)s',
            datefmt='%H:%M:%S',
        ))
        logger.addHandler(fh)
        self.logger = logger

        logger.info(f"cube        : {self.path_cube}")
        logger.info(f"classified  : {self.path_clfy}")
        logger.info(f"mask        : {self.path_mask}")
        logger.info(f"secondary   : {self.secondary_mode}")
        logger.info(f"bgsub       : {self.bgsub}")
        logger.info(f"std_channel : {self.std_channel:.4e}")
        logger.info("-" * 60)

    def log_summary(self) -> None:
        """Write a pixel-count / percentage summary to the logger.
        Call this after run().
        """
        if self.logger is None:
            return

        counts = Counter(rec[2] for rec in self._log_pixel_records)
        total  = sum(counts.values())

        self.logger.info("-" * 60)
        self.logger.info("STACKING SUMMARY")
        self.logger.info(f"  total pixels stacked : {total}")
        for method in ('baygaud', 'hermite', 'moment'):
            n   = counts.get(method, 0)
            pct = n / total * 100 if total > 0 else 0.0
            self.logger.info(f"  {method:<10}: {n:6d}  ({pct:5.1f}%)")
        self.logger.info("-" * 60)

        # Per-pixel table — only written if DEBUG is enabled
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("PIXEL LOG  (x, y, method, n_components)")
            for x, y, method, n_comp in self._log_pixel_records:
                self.logger.debug(f"  {x:4d}  {y:4d}  {method:<10}  {n_comp}")

    # ------------------------------------------------------------------

    def writestat(self, message: str) -> None:
        if self.path_temp is None:
            return
        try:
            (self.path_temp).mkdir(parents=True, exist_ok=True)
            with open(self.path_temp / f"stat.{self.name_cube}{self.suffix}.txt", "w") as f:
                f.write(f"{self.name_cube} {self.suffix} {message}")
        except Exception:
            pass

    def removestat(self) -> None:
        if self.path_temp is None:
            return
        path_stat = self.path_temp / f"stat.{self.name_cube}{self.suffix}.txt"
        if path_stat.exists():
            try:
                os.remove(path_stat)
            except Exception:
                pass
        
    def _mom0_to_NHI(self, mom0):
        NHI = (
            (mom0 * u.Jy).to(u.K, u.brightness_temperature(1.420 * u.GHz, self.beam)).value
            * 1e-3 * 1.823e18
        )
        return NHI
        
        
    def mask_specrange(self, dict_mask):
        """
        Mask (zero out) a range along the spectral axis of the cube.

        Parameters
        ----------
        low : astropy.units.Quantity | float | int | None
            Lower bound of the mask. Meaning depends on `mode`.
            - mode='velocity': km/s quantity (or numeric already in km/s)
            - mode='channel' : channel index (int, can be negative)
            If None, mask from the start up to `high`.
        high : astropy.units.Quantity | float | int | None
            Upper bound of the mask. Same unit/index semantics as `low`.
            If None, mask from `low` to the end.
        mode : {'velocity', 'channel'}
            How to interpret `low`/`high`.
            - 'velocity': compare against `self.spec_axis` (km/s)
            - 'channel' : compare against channel indices [0..N-1]

        Notes
        -----
        - Boundaries are *open* (low < x < high), matching your original code.
        - The spectral axis is assumed to be axis 0.
        - The selected range is set to 0 (masked); everything else is kept.
        """
        import astropy.units as u
        import numpy as np

        n_spec = self.data_cube.shape[0]
        
        fr   = dict_mask['from']
        to   = dict_mask['to']
        mask = dict_mask['mask']
        mode = dict_mask['mode']
        
        if mode == 'velocity':
            pass

        elif mode == 'channel':
            chans = np.arange(n_spec)
            if mask=='inner':
                mask = (chans > fr) & (chans < to)
            elif mask=='outer':
                mask = (chans < fr) | (chans > to)
        else:
            raise ValueError("mode must be 'velocity' or 'channel'")

        # Apply along spectral axis (assumed axis 0)
        mask_reshaped  = mask[:, np.newaxis, np.newaxis]
        self.data_cube = np.where(mask_reshaped, 0, self.data_cube)
        
    def filter_genuine(self, dict_G):
        
        nois   = dict_G['nois']
        velo   = dict_G['velo']
        bkgr   = dict_G['bkgr']
        disp   = dict_G['disp']
        e_disp = dict_G['e_disp']
        
        if disp<2*e_disp: return False
        
        model =  gauss(self.spec_axis, dict_G['ampl'], velo, disp)
        mask = (self.spec_axis>velo-3*disp)&(self.spec_axis<velo+3*disp)&\
               (model>2*nois+bkgr)

        if len(mask)>3:
            return True
        
        else: 
            return False
    
    def run(self, stack_secondary=False):
 
        data_cube = np.float64(self.data_cube.copy())
        data_mask = self.data_mask.copy()
 
        # ROLL ---------------------------------------------------
        len_specaxis = len(self.spec_axis)
        xx = np.arange(len_specaxis) - int((len_specaxis - 1) / 2)
        if self.chansep < 0:
            xx = xx[::-1]
        lenx = len(xx)
        shifter = shifter_gipsy
        # --------------------------------------------------------
 
        yy  = np.zeros_like(xx, dtype=np.float64)
        e_y = np.zeros_like(xx, dtype=np.float64)
        NN  = np.zeros_like(xx)
 
        sa_div_chan    = self.spec_axis / self.chansep
        index_centr    = np.argwhere(xx == 0).item()
        std_channel    = self.std_channel
        index_dict_stacked = 0
 
        clfy_available = self.path_clfy is not None and Path(self.path_clfy).exists()
 
        if clfy_available:
            index_dict_stacked = self._stack_baygaud(
                data_cube, data_mask, xx, yy, NN,
                shifter, index_centr, index_dict_stacked,
            )
 
        if stack_secondary:
            self._stack_secondary(
                data_cube, data_mask, xx, yy, NN,
                shifter, index_centr, index_dict_stacked,
            )
 
        # ---- finalise ------------------------------------------
        yy  = yy  * (u.Jy / u.beam)
        e_y = e_y * (u.Jy / u.beam)
 
        yy  =          yy.to(u.Jy / u.sr, equivalencies=u.beam_angular_area(self.beam)) * self.area_pixl
        e_y = std_channel.to(u.Jy / u.sr, equivalencies=u.beam_angular_area(self.beam)) * self.area_pixl * np.sqrt(NN / self.pix_per_beam)
        e_y[np.argwhere(e_y == 0)] = np.median(e_y)
 
        xx, yy, e_y = xx * np.abs(self.chansep), yy, e_y
 
        df = pd.DataFrame()
        df['x']   = xx
        df['y']   = yy.value
        df['e_y'] = e_y.value
        df['N']   = NN
 
        pixels_in_beam = self.beam.sr / (self.cd * self.cd).to(u.sr)
        df = df.loc[df['N'] > pixels_in_beam].reset_index(drop=True)
 
        self.xx  = df['x']
        self.yy  = df['y']
        self.e_y = df['e_y']
        self.df_stacked = df
 
    # ------------------------------------------------------------------
    def _stack_baygaud(
        self, data_cube, data_mask, xx, yy, NN,
        shifter, index_centr, index_dict_stacked,
    ):
        """Shift-and-stack pixels that have baygaud NGfit solutions."""
 
        dict_data = read_ngfits(
            self.path_clfy / 'ngfit',
            toreads=['flux', 'velo', 'disp', 'psnr', 'bkgr', 'nois', 'e_disp'],
            path_mask=self.path_mask,
            wo_unit=True,
        )
 
        if self.bgsub:
            for coord in dict_data.keys():
                for g in dict_data[coord].keys():
                    dict_data[coord][g]['bkgr'] = 0.
 
        for coord in dict_data.keys():
            for g in dict_data[coord].keys():
                if 'psnr' not in dict_data[coord][g]:
                    continue
                psnr = dict_data[coord][g]['psnr']
                nois = dict_data[coord][g]['nois']
                bkgr = dict_data[coord][g]['bkgr']
                dict_data[coord][g]['ampl'] = (psnr * nois) + bkgr
 
        len_data     = len(dict_data)
        count        = 0
        last_reported = -1
        self.stat    = 'STAKB'
 
        if self.pbar:
            pbar = tqdm(total=len_data)
 
        for coord in dict_data.keys():
            x, y = map(int, coord.split(','))
            dict_cord = dict_data[coord]
 
            data_mask[y, x] = np.nan
            self.map_stack_method[y, x] = 1
 
            all_models = {
                g: gauss(self.spec_axis, d['ampl'], d['velo'], d['disp'])
                for g, d in dict_cord.items()
                if 'velo' in d
            }
            model_sum = sum(all_models.values()) if all_models else 0.0
 
            for g in dict_cord:
                if 'velo' not in dict_cord[g]:
                    continue
 
                dict_pg    = dict_cord[g]
                data_subed = data_cube[:, y, x].copy()
                data_subed -= (model_sum - all_models[g])
                data_subed -= dict_pg['bkgr']
 
                shifted, valid = shifter(
                    data_subed,
                    self.spec_axis,
                    dict_pg['velo'],
                    xx * self.abschansep,
                    fill=0.0,
                    index_centr=index_centr,
                )
 
                yy += shifted
                NN += valid.astype(np.uint8)
 
                self.dict_stacked[index_dict_stacked] = {
                    'NHI':    self._mom0_to_NHI(dict_pg['flux']) * 1000,
                    'disp':   dict_pg['disp'],
                    'e_disp': dict_pg['e_disp'],
                }

                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"baygaud  ({x:4d},{y:4d})  g={g}  "
                        f"velo={dict_pg['velo']:8.2f} km/s  "
                        f"disp={dict_pg['disp']:6.2f} km/s  "
                        f"flux={dict_pg['flux']:.3e}"
                    )

                index_dict_stacked += 1

            # Record pixel after processing all components
            n_comp = len([g for g in dict_cord if 'velo' in dict_cord[g]])
            self._log_pixel_records.append((x, y, 'baygaud', n_comp))

            if self.pbar:
                pbar.update(1)
            count += 1
            prog_perc = count / len_data * 100
            if prog_perc // 10 > last_reported:
                last_reported = prog_perc // 10
                self.writestat(f"{self.stat} {prog_perc:.0f}% .")
 
        return index_dict_stacked
 
    # ------------------------------------------------------------------
    def _stack_secondary(
        self, data_cube, data_mask, xx, yy, NN,
        shifter, index_centr, index_dict_stacked,
    ):
        """Shift-and-stack pixels using the secondary velocity field
        (moment-1 or Hermite) for pixels not already consumed by baygaud."""
 
        path_mom0    = self.path_cube.parent / 'cube_mom0.fits'
        data_mom0    = fits.getdata(path_mom0) if path_mom0.exists() else None
 
        if self.secondary_mode in ('HER', 'HERMOM'):
            data_disp_sec = self.data_disp_secondary.copy()
            stat_label    = 'STAKHE'
        else:
            mom2_path = self.path_cube.parent / 'cube_mom2.fits'
            data_disp_sec = fits.getdata(mom2_path) / 1000. if mom2_path.exists() else None
            stat_label    = 'STAKM'
 
        data_vf_secondary = self.data_vf_secondary.copy()
        if data_disp_sec is not None:
            data_vf_secondary = np.where(
                (~np.isfinite(data_disp_sec)) | (data_disp_sec < self.chansep / 2.355 * 2),
                np.nan,
                data_vf_secondary,
            )
 
        valid_pixels = np.argwhere(
            np.isfinite(data_mask) & np.isfinite(data_vf_secondary)
        )
        len_data      = len(valid_pixels)
        count         = 0
        last_reported = -1
        self.stat     = stat_label

        if self.logger:
            self.logger.info(f"secondary mode : {self.secondary_mode}  ({len_data} pixels)")
 
        for y, x in valid_pixels:
 
            shifted, valid = shifter(
                data_cube[:, y, x],
                self.spec_axis,
                data_vf_secondary[y, x],
                xx * self.abschansep,
                fill=0.0,
                index_centr=index_centr,
            )
            yy += shifted
            NN += valid.astype(np.uint8)
 
            if self.secondary_mode == 'HER':
                method_code = 2
            elif self.secondary_mode == 'HERMOM':
                method_code = 2 if self._hermite_vf_finite[y, x] else 3
            else:
                method_code = 3
            self.map_stack_method[y, x] = method_code

            method_label = {1: 'baygaud', 2: 'hermite', 3: 'moment'}[method_code]
            self._log_pixel_records.append((x, y, method_label, 1))

            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{method_label:<8} ({x:4d},{y:4d})  "
                    f"vf={data_vf_secondary[y, x]:8.2f} km/s"
                )
 
            if data_disp_sec is not None:
                disp_val = data_disp_sec[y, x]
                if not np.isfinite(disp_val):
                    continue
                nhi = self._mom0_to_NHI(data_mom0[y, x]) if data_mom0 is not None else 0.0
                self.dict_stacked[index_dict_stacked] = {
                    'NHI':    nhi,
                    'disp':   disp_val,
                    'e_disp': 0.0,
                }
                index_dict_stacked += 1
 
            count += 1
            prog_perc = count / len_data * 100 if len_data > 0 else 100
            if prog_perc // 10 > last_reported:
                last_reported = prog_perc // 10
                self.writestat(f"{self.stat} {prog_perc:.0f}% .")
        
    def save_df_stacked(self, path_to_save):
        self.df_stacked.to_string(path_to_save, index=False)

    def plot_map_stack_method(self, ax=None, path_save=None):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        labels  = {1: 'baygaud', 2: 'hermite', 3: 'moment'}
        colors  = {0: 'white',   1: '#4477AA', 2: '#EE6677', 3: '#228833'}
        bounds  = [-0.5, 0.5, 1.5, 2.5, 3.5]
        cmap    = mcolors.ListedColormap([colors[k] for k in range(4)])
        norm    = mcolors.BoundaryNorm(bounds, cmap.N)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))

        m = self.map_stack_method.copy().astype(float)
        m[m == 0] = np.nan
        ax.imshow(m, origin='lower', cmap=cmap, norm=norm, interpolation='none')

        patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in (1, 2, 3)]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
        ax.set_title('Stacking method per pixel')
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')

        if standalone:
            plt.tight_layout()
            if path_save is not None:
                plt.savefig(path_save, dpi=150)
                plt.close(fig)
            else:
                plt.show()
    

    
if __name__=='__main__':
    
    multipliers = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    
    # for survey in ['LITTLE_THINGS','THINGS','VLA-ANGST','AVID','VIVA']:
    for survey in ['VIVA']:
    
        paths_cube = natsorted(list(Path(f'/home/mandu/workspace/research/data/{survey}_halfbeam').glob('*/cube.fits')))
        
        pbar = tqdm(paths_cube)
        
        for path_cube in pbar:
            
            name_cube = path_cube.parent.name
            path_clfy = path_cube.parent/'segmts_merged_n_classified.3'
            
            pbar.set_description(f'{survey},{name_cube}')
            
            df = pd.DataFrame()
            
            for i, multiplier in enumerate(multipliers):
                
                suffix = f'_I{multiplier}r25'
                
                path_mask = path_cube.parent/f'mask{suffix}.fits'
                
                sns = ShiftnStack(path_cube, path_clfy, path_mask=path_mask)
                sns.setup_logger(suffix=suffix)          # sets up log dir + file
                if name_cube=='NGC1569':
                    sns.mask_specrange(-20*(u.km/u.s), 20*(u.km/u.s))      
                sns.pbar = True
                sns.run()
                sns.log_summary()                        # writes summary to log
                
                df_stacked = sns.df_stacked
                df_stacked[suffix[1:]] = df_stacked['y']
                
                if i==0: df = df_stacked[['x',suffix[1:]]]
                else:
                    df = pd.merge(df, df_stacked[['x',suffix[1:]]])
            
            path_df_out = path_cube.parent/'df_stacked.csv'
            df.to_string(path_df_out, index=False)
                    
                
                
else:
    from .subroutines_shiftnstack import argfind_nearest, shifter