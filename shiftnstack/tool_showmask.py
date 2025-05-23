from pathlib import Path

import numpy as np
import pylab as plt
from astropy.io import fits

path_mask = Path('/home/mandu/workspace/research/data/AVID_halfbeam/VCC130/mask/mask_R0.5RHI.fits')

wdir = path_mask.parent.parent
# path_sgfit = wdir/'segmts_merged_n_classified.3/sgfit/sgfit.G3_1.1.fits'
path_sgfit = wdir/'cube_mom1.fits'


data_mask  = fits.getdata(path_mask)
data_sgfit = fits.getdata(path_sgfit)

data = np.where(np.isfinite(data_sgfit), data_mask, np.nan)

fig, axs = plt.subplots(ncols=2, sharex=True,sharey=True)

ax = axs[0]
ax.imshow(data_sgfit)

ax = axs[1]
ax.imshow(data)
ax.invert_yaxis()

plt.show()