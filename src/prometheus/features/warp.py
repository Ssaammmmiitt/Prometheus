"""Regrid any raster onto the canonical Nepal 1 km grid."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from prometheus import grid
from prometheus.config import load_settings

RESAMPLING = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "average": Resampling.average,
    "cubic": Resampling.cubic,
}


def warp_array(
    src: np.ndarray,
    src_transform: Affine,
    src_crs="EPSG:4326",
    *,
    resampling: str = "bilinear",
    dst_transform: Affine | None = None,
    dst_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reproject a 2D or 3D (band, y, x) array onto the canonical grid."""
    dst_transform = dst_transform if dst_transform is not None else grid.transform()
    dst_shape = dst_shape if dst_shape is not None else grid.shape()

    squeeze = src.ndim == 2
    stack = src[None, ...] if squeeze else src
    out = np.full((stack.shape[0], *dst_shape), np.nan, dtype=np.float32)

    reproject(
        source=np.ascontiguousarray(stack.astype(np.float32)),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=grid.crs(),
        dst_nodata=np.nan,
        resampling=RESAMPLING[resampling],
    )
    return out[0] if squeeze else out


def read_to_grid(
    path: Path | str,
    *,
    bands: list[int] | None = None,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, list[str]]:
    """Read a GeoTIFF and warp its bands onto the canonical grid."""
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        data = src.read(idx).astype(np.float32)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        names = [src.descriptions[i - 1] or f"band_{i}" for i in idx]
        warped = warp_array(
            data, src.transform, src.crs, resampling=resampling
        )
    return warped, names


def gee_static_dir() -> Path:
    return load_settings().paths.resolve("gee_raw") / "static"


@lru_cache(maxsize=1)
def elevation_1km() -> np.ndarray:
    """Canonical-grid elevation (m). Prefers the aligned local SRTM raster."""
    settings = load_settings()
    local = settings.paths.resolve("elevation")
    if local.is_file():
        with rasterio.open(local) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            if src.shape == grid.shape():
                return arr
            return warp_array(arr, src.transform, src.crs)
    esa = gee_static_dir() / "elev_slope_aspect.tif"
    if not esa.is_file():
        raise FileNotFoundError(f"No elevation raster: {local} or {esa}")
    data, _ = read_to_grid(esa, bands=[1])
    return data


@lru_cache(maxsize=8)
def coarse_elevation(
    transform_key: tuple[float, ...],
    shape_key: tuple[int, int],
) -> np.ndarray:
    """
    Elevation of a coarse (e.g. ERA5) cell, evaluated back on the 1 km grid.

    Block-average 1 km SRTM up to the coarse grid, then bilinearly resample it
    back down. This mirrors how the coarse field itself is interpolated, so the
    lapse-rate delta only carries the sub-grid terrain that ERA5 cannot see.
    """
    src_transform = Affine(*transform_key)
    fine = elevation_1km()
    fine = np.where(np.isfinite(fine), fine, np.nan)

    coarse = np.full(shape_key, np.nan, dtype=np.float32)
    reproject(
        source=fine,
        destination=coarse,
        src_transform=grid.transform(),
        src_crs=grid.crs(),
        src_nodata=np.nan,
        dst_transform=src_transform,
        dst_crs=grid.crs(),
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    coarse = fill_nan_nearest(coarse)
    return warp_array(coarse, src_transform, grid.crs(), resampling="bilinear")


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs with the value of the nearest finite pixel."""
    from scipy import ndimage

    bad = ~np.isfinite(arr)
    if not bad.any() or bad.all():
        return arr
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


__all__ = [
    "coarse_elevation",
    "elevation_1km",
    "fill_nan_nearest",
    "gee_static_dir",
    "read_to_grid",
    "warp_array",
]
