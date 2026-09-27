"""
Georeferenced raster I/O for satsr.

The whole point of production remote-sensing SR is that the output stays on the
map: same CRS, correct (finer) geotransform, valid nodata. This module wraps
rasterio to guarantee that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


@dataclass
class RasterInfo:
    width: int
    height: int
    count: int
    dtype: str
    crs: object
    transform: Affine
    nodata: Optional[float]

    @property
    def res(self) -> Tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))


def read_info(path: str) -> RasterInfo:
    with rasterio.open(path) as ds:
        return RasterInfo(ds.width, ds.height, ds.count, ds.dtypes[0],
                          ds.crs, ds.transform, ds.nodata)


def read_window(path: str, row0: int, col0: int, h: int, w: int) -> np.ndarray:
    """Read a (h,w,bands) window; out-of-range is clipped. Returns native dtype."""
    with rasterio.open(path) as ds:
        h = min(h, ds.height - row0)
        w = min(w, ds.width - col0)
        arr = ds.read(window=Window(col0, row0, w, h))    # (bands,h,w)
    return np.transpose(arr, (1, 2, 0))


def read_all(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    return np.transpose(arr, (1, 2, 0))


def read_overview(path: str, max_dim: int = 384) -> np.ndarray:
    """Read a decimated (h,w,bands) overview of the whole scene (cheap)."""
    with rasterio.open(path) as ds:
        dec = max(1, max(ds.width, ds.height) // max_dim)
        arr = ds.read(out_shape=(ds.count, max(1, ds.height // dec), max(1, ds.width // dec)))
    return np.transpose(arr, (1, 2, 0))


def scaled_transform(transform: Affine, scale: float) -> Affine:
    """Shrink pixel size by `scale` (more, finer pixels); origin unchanged."""
    return transform * Affine.scale(1.0 / scale, 1.0 / scale)


def create_writer(path: str, info: RasterInfo, scale: float, count: int,
                  dtype: str = "uint16", compress: str = "deflate",
                  nodata: Optional[float] = None, block: int = 512):
    """
    Open a GeoTIFF for writing at `scale`x finer resolution than `info`.
    Preserves CRS; rescales the transform; tiled + compressed for large scenes.
    Returns an open rasterio dataset (caller must close).
    """
    out_h = int(round(info.height * scale))
    out_w = int(round(info.width * scale))
    profile = {
        "driver": "GTiff",
        "width": out_w,
        "height": out_h,
        "count": count,
        "dtype": dtype,
        "crs": info.crs,
        "transform": scaled_transform(info.transform, scale),
        "compress": compress,
        "tiled": True,
        "blockxsize": block,
        "blockysize": block,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    return rasterio.open(path, "w", **profile), (out_h, out_w)


def write_window(ds, array_hwc: np.ndarray, row0: int, col0: int) -> None:
    """Write a (h,w,bands) array block into an open writer at (row0,col0)."""
    arr = np.transpose(array_hwc, (2, 0, 1))    # (bands,h,w)
    ds.write(arr, window=Window(col0, row0, arr.shape[2], arr.shape[1]))
