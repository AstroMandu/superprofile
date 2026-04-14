import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def _read_cube_and_speccoords_increasing(path_cube, squeeze=True):
    """
    Read a FITS cube and derive spectral coordinate array via WCS.
    Ensures spec_coords is strictly increasing by reversing cube/spec if needed.
    Returns cube as (nz, ny, nx), spec_coords (nz,) increasing.
    """
    with fits.open(path_cube, memmap=True) as hdul:
        hdu = hdul[0]
        data = hdu.data
        header = hdu.header

    if data is None:
        raise ValueError("No data found in FITS HDU[0].")

    if squeeze:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D cube after squeeze; got shape {data.shape}")

    data = data.astype(np.float32, copy=False)

    w = WCS(header)

    # Find spectral WCS axis
    spec_wcs_axis = None
    for i, ctype in enumerate(w.wcs.ctype):
        if not ctype:
            continue
        cu = ctype.upper()
        if ("FREQ" in cu) or ("VRAD" in cu) or ("VELO" in cu) or ("VOPT" in cu) or ("WAVE" in cu):
            spec_wcs_axis = i
            break
    if spec_wcs_axis is None:
        raise ValueError(f"Could not identify a spectral axis from WCS CTYPE={w.wcs.ctype}")

    nz, ny, nx = data.shape
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0

    xpix = np.full(nz, x0, dtype=float)
    ypix = np.full(nz, y0, dtype=float)
    zpix = np.arange(nz, dtype=float)

    world = w.all_pix2world(xpix, ypix, zpix, 0)
    spec_coords = np.asarray(world[spec_wcs_axis], dtype=np.float64)

    # Ensure increasing (handle negative CDELT3)
    if nz >= 2 and spec_coords[1] < spec_coords[0]:
        spec_coords = spec_coords[::-1].copy()
        data = data[::-1, :, :].copy()

    return data, spec_coords


def shuffle_like_gipsy(
    path_cube: str,
    centers: np.ndarray,
    nmax: int,
    cdelt: float | None = None,
    blank: float = np.nan,
    squeeze: bool = True,
    dtype=np.float32,
):
    """
    Fast SHUFFLE-like resampling from FITS path.

    Preserves SHUFFLE logic:
      - adjacent spectral plane interpolation
      - output offsets l=-nmax..+nmax
      - fill only if blank
      - skip pixels where cn/i1/i2 are blank

    Parameters
    ----------
    path_cube : str
        FITS cube path.
    centers : (ny, nx) array
        Per-pixel center coordinate in SAME units as the cube spectral world coord.
        Use np.nan for blank.
    nmax : int
        Half-width of output offset axis in bins (total 2*nmax+1).
    cdelt : float, optional
        Output bin spacing (positive). If None, uses SHUFFLE default based on spectral span.
    blank : float
        Output blank value (np.nan recommended).
    squeeze : bool
        Whether to squeeze singleton axes in FITS data.
    dtype : numpy dtype
        Output dtype (float32 is faster/lighter).

    Returns
    -------
    out : (2*nmax+1, ny, nx) array
    out_offsets : (2*nmax+1,) array
    spec_coords : (nz,) array  (increasing)
    """
    cube, spec_coords = _read_cube_and_speccoords_increasing(path_cube, squeeze=squeeze)
    cube = cube.astype(dtype, copy=False)

    nz, ny, nx = cube.shape
    centers = np.asarray(centers, dtype=dtype)
    if centers.shape != (ny, nx):
        raise ValueError(f"centers shape {centers.shape} must match (ny,nx)=({ny},{nx}) from cube.")

    # Choose cdelt (always positive)
    if cdelt is None:
        cdelt = (spec_coords[-1] - spec_coords[0]) / (2.0 * nmax)
    cdelt = float(abs(cdelt))
    if cdelt <= 0:
        raise ValueError("cdelt must be positive.")

    # Output
    out = np.full((2 * nmax + 1, ny, nx), blank, dtype=dtype)
    lvals = np.arange(-nmax, nmax + 1, dtype=np.int32)
    out_offsets = (lvals * cdelt).astype(np.float64)

    # Flatten spatial dims
    npix = ny * nx
    cn = centers.reshape(npix)
    cn_ok = np.isfinite(cn)

    # Main loop over adjacent planes
    for iz in range(1, nz):
        c1 = float(spec_coords[iz - 1])
        c2 = float(spec_coords[iz])
        if c2 == c1:
            continue

        i1 = cube[iz - 1].reshape(npix)
        i2 = cube[iz].reshape(npix)

        ok = cn_ok & np.isfinite(i1) & np.isfinite(i2)
        if not np.any(ok):
            continue

        # Compute ls/le for ok pixels
        ls = np.empty(npix, dtype=np.int32)
        le = np.empty(npix, dtype=np.int32)
        ls.fill(1)
        le.fill(0)

        tmp_ls = np.ceil((c1 - cn[ok].astype(np.float64)) / cdelt).astype(np.int32)
        tmp_le = np.floor((c2 - cn[ok].astype(np.float64)) / cdelt).astype(np.int32)

        tmp_ls = np.maximum(tmp_ls, -nmax)
        tmp_le = np.minimum(tmp_le, +nmax)

        ls[ok] = tmp_ls
        le[ok] = tmp_le

        active = ok & (le >= ls)
        if not np.any(active):
            continue

        lmin = int(ls[active].min())
        lmax = int(le[active].max())

        denom = (c2 - c1)

        for l in range(lmin, lmax + 1):
            k = l + nmax
            out_k = out[k].reshape(npix)

            # Eligible pixels at this l and still blank
            m = active & (ls <= l) & (le >= l) & (~np.isfinite(out_k))
            if not np.any(m):
                continue

            c = cn[m].astype(np.float64) + (l * cdelt)
            w = (c2 - c) / denom

            out_k[m] = (i1[m] * w + i2[m] * (1.0 - w)).astype(dtype)
            out[k] = out_k.reshape(ny, nx)

    return out, out_offsets, spec_coords
