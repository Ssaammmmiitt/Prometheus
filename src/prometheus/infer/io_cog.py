"""Write risk maps as Cloud-Optimised (tiled, overview) GeoTIFFs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from prometheus import grid

# Risk is a probability; use a dedicated nodata so 0.0 remains a valid score.
RISK_NODATA = -1.0


def forecasts_dir() -> Path:
    from prometheus.config import load_settings

    path = load_settings().paths.resolve("runs") / "forecasts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def risk_path(when, horizon: int, root: Path | None = None) -> Path:
    day = str(when)[:10]
    return (root or forecasts_dir()) / f"risk_{day}_h{horizon}.tif"


def districts_path(when, root: Path | None = None) -> Path:
    day = str(when)[:10]
    return (root or forecasts_dir()) / f"districts_{day}.geojson"


def write_risk_cog(array: np.ndarray, path: Path) -> Path:
    """
    Write a single-band risk surface as a QGIS-friendly COG-like GeoTIFF.

    Tiled blocks + internal overviews is what Desktop GIS and TiTiler both want;
    True Cloud-Optimized GeoTIFF is the same layout (the COG driver is optional
    and not present in every rasterio wheel, so this path is pure rasterio).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array, dtype=np.float32)
    if data.shape != grid.shape():
        raise ValueError(f"risk shape {data.shape} != grid {grid.shape()}")

    written = np.where(np.isfinite(data), data, RISK_NODATA).astype(np.float32)
    profile = grid.profile(dtype="float32", nodata_value=RISK_NODATA)
    profile.update(
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="band",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(written, 1)
        dst.set_band_description(1, "fire risk probability (calibrated)")
        dst.update_tags(
            AREA_OR_POINT="Area",
            PROMETHEUS_VAR="risk",
            PROMETHEUS_UNITS="probability",
        )
        factors = [2, 4, 8, 16]
        dst.build_overviews(factors, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    return path


def read_risk(path: Path) -> np.ndarray:
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else RISK_NODATA
    return np.where(data == nodata, np.nan, data)


def is_complete(when, *, root: Path | None = None, horizons: list[int] = (1, 7)) -> bool:
    """True when all risk COGs and the district GeoJSON already exist and are non-empty."""
    root = root or forecasts_dir()
    paths = [risk_path(when, h, root) for h in horizons] + [districts_path(when, root)]
    return all(p.is_file() and p.stat().st_size > 0 for p in paths)


__all__ = [
    "RISK_NODATA",
    "districts_path",
    "forecasts_dir",
    "is_complete",
    "read_risk",
    "risk_path",
    "write_risk_cog",
]
