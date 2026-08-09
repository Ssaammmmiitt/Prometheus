"""Canonical Nepal 1 km grid + mask. Defined once; all rasters use this."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from prometheus.config import cfg, load_settings


def transform() -> Affine:
    """Rasterio/GDAL affine transform for the study grid."""
    t = load_settings().grid.transform
    if len(t) != 6:
        raise ValueError(f"grid.transform must have 6 elements, got {len(t)}")
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def crs() -> CRS:
    return CRS.from_string(load_settings().grid.crs)


def shape() -> tuple[int, int]:
    g = load_settings().grid
    return (g.height, g.width)


def height() -> int:
    return load_settings().grid.height


def width() -> int:
    return load_settings().grid.width


def nodata() -> float:
    return load_settings().grid.nodata


def mask_path() -> Path:
    return load_settings().paths.resolve("nepal_mask")


@lru_cache(maxsize=1)
def nepal_mask() -> np.ndarray:
    """
    Boolean mask of valid Nepal land pixels on the canonical grid.

    Shape (H, W). True inside Nepal (mask band == valid value).
    """
    import rasterio

    path = mask_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Nepal mask not found at {path}. "
            "Place nepal_mask_1km_roiAligned.tif there before using the grid."
        )

    settings = load_settings()
    with rasterio.open(path) as src:
        if src.shape != settings.grid.shape:
            raise ValueError(
                f"Mask shape {src.shape} != config grid {settings.grid.shape}"
            )
        band = src.read(1)
        valid = int(settings.grid.mask_valid_value)
        return band == valid


def profile(dtype: str = "float32", count: int = 1, nodata_value: float | None = None) -> dict:
    """Base rasterio profile for writing aligned GeoTIFFs."""
    settings = load_settings()
    nd = settings.grid.nodata if nodata_value is None else nodata_value
    return {
        "driver": "GTiff",
        "height": settings.grid.height,
        "width": settings.grid.width,
        "count": count,
        "dtype": dtype,
        "crs": settings.grid.crs,
        "transform": transform(),
        "nodata": nd,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def assert_aligned(path: Path | str) -> None:
    """Raise if a GeoTIFF is not on the canonical grid."""
    import rasterio

    path = Path(path)
    settings = load_settings()
    with rasterio.open(path) as src:
        if src.shape != settings.grid.shape:
            raise AssertionError(f"{path.name}: shape {src.shape} != {settings.grid.shape}")
        if src.crs is None or src.crs.to_string() not in (
            settings.grid.crs,
            "EPSG:4326",
            "+init=epsg:4326",
        ):
            # tolerate EPSG string form differences if same authority code
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise AssertionError(f"{path.name}: CRS {src.crs} != {settings.grid.crs}")
        st = transform()
        for a, b in zip(src.transform[:6], st[:6]):
            if abs(float(a) - float(b)) > 1e-9:
                raise AssertionError(f"{path.name}: transform mismatch {src.transform} vs {st}")


def n_valid_pixels() -> int:
    return int(nepal_mask().sum())


# Re-export convenience used by callers
__all__ = [
    "assert_aligned",
    "crs",
    "height",
    "mask_path",
    "n_valid_pixels",
    "nepal_mask",
    "nodata",
    "profile",
    "shape",
    "transform",
    "width",
    "cfg",
]
