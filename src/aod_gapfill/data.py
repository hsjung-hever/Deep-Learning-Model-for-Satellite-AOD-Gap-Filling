from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr


RNG_SEED = 731


def minmax_norm(dt: xr.DataArray):
    vmax = float(dt.max().values)
    vmin = float(dt.min().values)
    normed = (dt.astype("float32") - vmin) / (vmax - vmin + 1e-12)
    return normed, vmin, vmax


def to_img4(ds: xr.Dataset, var: str, sz: int):
    return ds[var].values.reshape((-1, sz, sz, 1)).astype("float32")


def split_indices(n: int, test_ratio: float, seed: int = RNG_SEED):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    testN = int(n * test_ratio)
    tests = set(rng.choice(idx, size=testN, replace=False).tolist())
    trains = [int(i) for i in idx if int(i) not in tests]
    return trains, list(tests)


def make_gapped(imgs: np.ndarray, gap_size: int, miss_val: float = 0.0):
    out = imgs.copy()
    N, H, W, _ = out.shape
    rng = np.random.default_rng(RNG_SEED)
    rn = int(gap_size)
    for i in range(N):
        bound = H - rn - 1
        ix = int(rng.integers(0, bound))
        iz = int(rng.integers(0, bound))
        ic = int(rng.integers(0, bound))
        out[i, 10:20, ix:ix+rn, 0] = miss_val
        out[i, iz:iz+rn, 5:14, 0] = miss_val
        out[i, ic:ic+rn, ic:ic+rn, 0] = miss_val
    return out


def load_and_normalize(nc_path: str, sz: int, varnames: list[str]):
    ds = xr.open_dataset(nc_path, engine="scipy")
    dsnorm = ds.copy()
    minmax = {}
    for v in varnames:
        if v in dsnorm:
            dsnorm[v], vmin, vmax = minmax_norm(dsnorm[v])
            minmax[v] = (vmin, vmax)
    return dsnorm, minmax