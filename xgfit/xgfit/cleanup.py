# Expects on self: path_plot, name_cube, suffix, list_disp, list_NHI_,
#                  df, df_params, gmodel, gmodel_1G, sampler, resampled,
#                  burnin, thin
from __future__ import annotations

import ctypes
import gc

import matplotlib.pyplot as plt


class CleanupMixin:
    

    def cleanup(self, *, aggressive: bool = True, trim_malloc: bool = True) -> None:
        """Best-effort memory cleanup for long batch runs."""
        # 1) Close matplotlib figures (this is big)
        try:
            plt.close("all")
        except Exception:
            pass

        # 2) Delete / null big attributes (only those you don't need afterwards)
        big_attrs = [
            # raw data arrays
            "x", "y", "e_y",
            "resampled", "flat_samples",

            # models can hold references to data arrays
            "gmodel", "gmodel_1G",

            # sometimes large
            "df_stacked",

            # any cached dicts that might be large
            "dict_stacked", "list_disp","sampler"
        ]

        for a in big_attrs:
            if hasattr(self, a):
                try:
                    setattr(self, a, None)
                except Exception:
                    pass

        # 3) If you created any very large local caches, clear them too
        # Example: autocorr buffers
        for a in ["autocorr_mean", "autocorr_max", "chekstep"]:
            if hasattr(self, a):
                try:
                    setattr(self, a, None)
                except Exception:
                    pass

        # 4) Force garbage collection
        gc.collect()

        # 5) Optional: on Linux, ask libc to return freed arenas to OS
        # This can reduce RSS *sometimes*.
        if aggressive and trim_malloc:
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass