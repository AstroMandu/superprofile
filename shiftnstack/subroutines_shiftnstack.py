import numpy as np
from numba import njit


def select_spatial_mask(data, value=None):
    """Convert a mask to 1/NaN, retaining NaN exclusions for either IO choice."""
    if value is None:
        return data
    if value not in (0, 1):
        raise ValueError('mask_value must be 0 (outer), 1 (inner), or None')
    return np.where(np.isfinite(data) & (data == value), 1.0, np.nan)


@njit(fastmath=True, cache=True)
def argfind_nearest(array, value):
    n = array.shape[0]
    idx = np.searchsorted(array, value)
    if idx == 0: return 0
    if idx >= n: return n - 1
    return idx-1 if abs(value - array[idx-1]) <= abs(value - array[idx]) else idx
    
def shifter(data, mod, lenx, *, dtype=None):
    if dtype is None:
        dtype = data.dtype
    shifted = np.zeros(lenx, dtype=dtype)
    # if mod >= lenx or mod + 1 <= 0:
    #     print(f'[Shiftnstack] mod length out of range! mod: {mod} lenx: {lenx}')
    #     return shifted  # completely out of range or negative far left
    # # compute overlap
    
    start_dst = max(mod, 0)
    end_dst   = min(mod + data.shape[0], lenx)
    start_src = start_dst - mod
    end_src   = start_src + (end_dst - start_dst)
    if end_dst > start_dst:
        shifted[start_dst:end_dst] = data[start_src:end_src]
    return shifted
    
    # shifted = np.roll(data, mod)
    
    return shifted

def shifter_roll(data, spec_axis, center, xx, fill=None, index_centr=None):
    
    if spec_axis[1]<spec_axis[0]:
        data = data[::-1]
        spec_axis = spec_axis[::-1]
    
    index = argfind_nearest(spec_axis, center)
    mod   = index_centr - index

    shifted = np.roll(data, mod)
    
    return shifted, np.ones(len(shifted))


@njit(fastmath=True, cache=True)
def shifter_gipsy(yy, sa, center, xx, fill=0.0, index_centr=0):
    if sa[1] < sa[0]:
        sa = sa[::-1]
        yy = yy[::-1]

    n       = sa.shape[0]
    m       = xx.shape[0]
    shifted = np.empty(m, dtype=np.float64)
    valid   = np.empty(m, dtype=np.bool_)

    sa0 = sa[0]
    sa1 = sa[n-1]

    for i in range(m):
        t = center + xx[i]
        if t < sa0 or t > sa1:
            shifted[i] = fill
            valid[i]   = False
        else:
            # binary search
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if sa[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            # linear interpolation
            dt = (t - sa[lo]) / (sa[hi] - sa[lo])
            shifted[i] = yy[lo] + dt * (yy[hi] - yy[lo])
            valid[i]   = True

    return shifted, valid


def summed_flux_error(noise_per_beam, counts, pixels_per_beam):
    """Approximate Jy error of a spatial sum, treating beam areas as independent.

    N pixels cover N/P beams. The integrated-flux uncertainty is therefore
    sigma_beam * sqrt(N/P), without an additional pixel/beam conversion.
    Interpolation and full noise covariance are not modeled here.
    """
    return noise_per_beam * np.sqrt(np.asarray(counts) / pixels_per_beam)
